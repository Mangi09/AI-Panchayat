/**
 * AI Panchayat Dashboard – App.jsx  (BUGFIX v2.1)
 *
 * Bug 1 fix: removed manual headers: { 'Content-Type': 'multipart/form-data' }
 *   from EVERY axios call that sends FormData.  When you set Content-Type
 *   manually, axios/fetch strips the required `boundary` parameter that
 *   separates multipart fields.  FastAPI's parser cannot find the `file`
 *   field without the boundary → immediate 400.  Deleting the headers block
 *   lets the browser set the full header including boundary automatically.
 *
 * Bug 2 fix: all catch blocks now extract err.response?.data?.detail before
 *   falling back to err.message.  FastAPI always puts the human-readable
 *   description in response.data.detail; err.message is always the generic
 *   "Request failed with status code 400" which is useless to the user.
 *
 * Bug 4 fix (UI side): added a COLUMN_PICKING state between file-drop and
 *   analysis.  After a file is selected, the frontend calls POST /api/columns
 *   to get the column list, then shows the user two <select> dropdowns.
 *   The chosen column names are sent as URL query params to /api/audit
 *   and /api/mitigate.  No external CSV will ever guess the wrong column
 *   again.
 */

import React, { useState } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';
import {
  UploadCloud, FileText, AlertTriangle, ShieldCheck, Scale,
  CheckCircle, RefreshCw, Zap, TrendingDown, TrendingUp,
  ArrowRight, Info, Columns,
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

// ── Shared helpers ────────────────────────────────────────────────────────────

/** BUG 2 FIX: always surface the FastAPI detail string, not the generic axios message */
function extractError(err) {
  return (
    err?.response?.data?.detail ||
    (typeof err?.response?.data === 'string' ? err.response.data : null) ||
    err?.message ||
    'An unknown error occurred'
  );
}

// ── Mitigation method metadata ───────────────────────────────────────────────
const MITIGATION_METHODS = [
  {
    key: 'reweighing',
    label: 'Reweighing',
    badge: 'Pre-processing',
    badgeColor: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
    description: 'Re-weights training samples so each (group × label) cell contributes equally to the loss. Zero architectural change.',
    icon: '⚖️',
  },
  {
    key: 'exponentiated_gradient',
    label: 'Exponentiated Gradient',
    badge: 'In-processing',
    badgeColor: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    description: 'Converts the fairness constraint into a Lagrangian saddle-point problem, training a mixture of classifiers that satisfies Demographic Parity.',
    icon: '📐',
  },
  {
    key: 'threshold_optimizer',
    label: 'Threshold Optimizer',
    badge: 'Post-processing',
    badgeColor: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
    description: 'Finds per-group decision thresholds on the ROC curve to equalise positive-prediction rates — no retraining required.',
    icon: '🎯',
  },
];

// ── Sub-components ────────────────────────────────────────────────────────────

function MetricDelta({ label, before, after, higherIsBetter = true }) {
  const improved = higherIsBetter ? after >= before : after <= before;
  const delta = ((after - before) * 100).toFixed(2);
  const sign = delta > 0 ? '+' : '';
  return (
    <div className="flex items-center justify-between p-3 rounded-xl bg-gray-800/50 border border-gray-700/50">
      <span className="text-sm text-gray-400">{label}</span>
      <div className="flex items-center gap-3">
        <span className="text-gray-500 text-sm">{(before * 100).toFixed(1)}%</span>
        <ArrowRight className="w-3 h-3 text-gray-600" />
        <span className="text-white font-semibold text-sm">{(after * 100).toFixed(1)}%</span>
        <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${improved ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>
          {sign}{delta}pp
        </span>
      </div>
    </div>
  );
}

function MitigationResultPanel({ result, onReset }) {
  const { metrics, simulation, csv_string } = result;
  const { original, mitigated, improvement } = metrics;
  
  const handleDownload = () => {
    const blob = new Blob([csv_string], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'unbiased_dataset_certified.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const getAgentColor = (a) => {
    if (a.includes('Data Scientist')) return 'text-blue-400 bg-blue-400/10 border-blue-400/30';
    if (a.includes('Ethics'))         return 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30';
    if (a.includes('Legal') || a.includes('Compliance')) return 'text-rose-400 bg-rose-400/10 border-rose-400/30';
    return 'text-purple-400 bg-purple-400/10 border-purple-400/30';
  };
  const getAgentIcon = (a) => {
    if (a.includes('Data Scientist')) return <FileText   className="w-5 h-5 text-blue-400" />;
    if (a.includes('Ethics'))         return <ShieldCheck className="w-5 h-5 text-emerald-400" />;
    if (a.includes('Legal') || a.includes('Compliance')) return <Scale className="w-5 h-5 text-rose-400" />;
    return <AlertTriangle className="w-5 h-5 text-purple-400" />;
  };

  const groupRateData = Object.keys(original.group_acceptance_rates).map((group) => ({
    group,
    Before: +(original.group_acceptance_rates[group] * 100).toFixed(1),
    After:  +(mitigated.group_acceptance_rates[group] * 100).toFixed(1),
  }));

  const radarData = [
    { metric: 'Accuracy',  Before: +(original.model_metrics.accuracy * 100).toFixed(1),  After: +(mitigated.model_metrics.accuracy * 100).toFixed(1) },
    { metric: 'Precision', Before: +(original.model_metrics.precision * 100).toFixed(1), After: +(mitigated.model_metrics.precision * 100).toFixed(1) },
    { metric: 'Recall',    Before: +(original.model_metrics.recall * 100).toFixed(1),    After: +(mitigated.model_metrics.recall * 100).toFixed(1) },
    { metric: 'F1',        Before: +(original.model_metrics.f1_score * 100).toFixed(1),  After: +(mitigated.model_metrics.f1_score * 100).toFixed(1) },
    {
      metric: 'Fairness (inv. DP)',
      Before: +(Math.max(0, 1 - Math.abs(original.bias_metrics.demographic_parity_difference)) * 100).toFixed(1),
      After:  +(Math.max(0, 1 - Math.abs(mitigated.bias_metrics.demographic_parity_difference)) * 100).toFixed(1),
    },
  ];

  const dpBefore = Math.abs(original.bias_metrics.demographic_parity_difference);
  const dpAfter  = Math.abs(mitigated.bias_metrics.demographic_parity_difference);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-emerald-300 flex items-center gap-2">
          <CheckCircle className="w-6 h-6" /> Mitigation Applied
        </h2>
        <button onClick={onReset} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-800/80 hover:bg-gray-700 transition text-sm">
          <RefreshCw className="w-4 h-4" /> New Audit
        </button>
      </div>

      <div className="glass-dark rounded-2xl p-4 border border-emerald-500/20 bg-emerald-900/10 flex items-start gap-3">
        <Info className="w-5 h-5 text-emerald-400 mt-0.5 shrink-0" />
        <div>
          <p className="text-emerald-300 font-semibold text-sm">{improvement.method}</p>
          <p className="text-emerald-100/70 text-sm mt-1">{improvement.description}</p>
          <p className="text-yellow-300/70 text-xs mt-1">Trade-off: {improvement.trade_off}</p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'DP Bias Reduced',   value: `${improvement.dp_reduction_pct}%`,  icon: <TrendingDown className="w-5 h-5" />, color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/20' },
          { label: 'EO Bias Reduced',   value: `${improvement.eo_reduction_pct}%`,  icon: <TrendingDown className="w-5 h-5" />, color: 'text-cyan-400',    bg: 'bg-cyan-500/10 border-cyan-500/20' },
          {
            label: 'Accuracy Δ',
            value: `${improvement.accuracy_change_pct > 0 ? '+' : ''}${improvement.accuracy_change_pct}pp`,
            icon: improvement.accuracy_change_pct >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />,
            color: improvement.accuracy_change_pct >= -2 ? 'text-yellow-300' : 'text-rose-400',
            bg: 'bg-gray-800/60 border-gray-700/50',
          },
        ].map((kpi) => (
          <div key={kpi.label} className={`glass-dark rounded-2xl p-5 border ${kpi.bg} flex flex-col items-center text-center gap-1`}>
            <span className={kpi.color}>{kpi.icon}</span>
            <span className={`text-3xl font-black ${kpi.color}`}>{kpi.value}</span>
            <span className="text-gray-400 text-xs">{kpi.label}</span>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-dark rounded-3xl p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-200">Group Acceptance Rate: Before vs After</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={groupRateData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
              <XAxis dataKey="group" stroke="#9ca3af" tick={{ fontSize: 12 }} />
              <YAxis stroke="#9ca3af" tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
              <Tooltip cursor={{ fill: 'rgba(255,255,255,0.04)' }} contentStyle={{ backgroundColor: 'rgba(17,24,39,0.95)', borderColor: '#374151', borderRadius: '8px' }} formatter={(v) => [`${v}%`]} />
              <Legend />
              <Bar dataKey="Before" fill="#ef4444" radius={[4, 4, 0, 0]} />
              <Bar dataKey="After"  fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-dark rounded-3xl p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-200">Model Metrics Radar</h3>
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <PolarRadiusAxis domain={[0, 100]} tick={{ fill: '#6b7280', fontSize: 10 }} />
              <Radar name="Before" dataKey="Before" stroke="#ef4444" fill="#ef4444" fillOpacity={0.15} />
              <Radar name="After"  dataKey="After"  stroke="#10b981" fill="#10b981" fillOpacity={0.15} />
              <Legend />
              <Tooltip contentStyle={{ backgroundColor: 'rgba(17,24,39,0.95)', borderColor: '#374151', borderRadius: '8px' }} formatter={(v) => [`${v}%`]} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-dark rounded-3xl p-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-200">Detailed Metric Comparison</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <MetricDelta label="Accuracy"  before={original.model_metrics.accuracy}  after={mitigated.model_metrics.accuracy} />
          <MetricDelta label="Precision" before={original.model_metrics.precision} after={mitigated.model_metrics.precision} />
          <MetricDelta label="Recall"    before={original.model_metrics.recall}    after={mitigated.model_metrics.recall} />
          <MetricDelta label="F1 Score"  before={original.model_metrics.f1_score}  after={mitigated.model_metrics.f1_score} />
          <MetricDelta label="Demographic Parity Diff (↓ better)" before={dpBefore} after={dpAfter} higherIsBetter={false} />
          <MetricDelta label="Equalized Odds Diff (↓ better)" before={Math.abs(original.bias_metrics.equalized_odds_difference)} after={Math.abs(mitigated.bias_metrics.equalized_odds_difference)} higherIsBetter={false} />
        </div>
      </div>

      {simulation && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          <div className="glass-dark rounded-3xl p-6 flex flex-col max-h-[600px]">
            <h3 className="text-xl font-semibold mb-4 border-b border-gray-700/50 pb-4 flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-emerald-400" /> MiroFish Mitigation Debate
            </h3>
            <div className="flex-1 overflow-y-auto pr-2 space-y-4 custom-scrollbar">
              {simulation.mitigation_debate?.map((turn, i) => (
                <div key={i} className={`flex flex-col p-4 rounded-2xl border ${getAgentColor(turn.agent)}`}>
                  <div className="flex items-center gap-2 mb-2">
                    {getAgentIcon(turn.agent)}
                    <span className="font-bold text-sm tracking-wide uppercase">{turn.agent}</span>
                  </div>
                  <p className="text-gray-200 text-sm leading-relaxed">{turn.dialogue}</p>
                </div>
              ))}
            </div>
          </div>
          <div className="glass-dark rounded-3xl p-8 relative overflow-hidden bg-emerald-900/20 border border-emerald-500/40">
            <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none" />
            <h3 className="text-2xl font-bold mb-6 flex items-center gap-3 text-emerald-300">
              <CheckCircle className="w-6 h-6 text-emerald-400" /> Certified Unbiased Report
            </h3>
            <div className="space-y-4 mb-8">
              <div className="p-4 rounded-xl bg-emerald-950/50 border border-emerald-500/20">
                <span className="text-xs text-emerald-500 font-bold uppercase tracking-wider block mb-1">Status</span>
                <span className="text-lg text-emerald-100">{simulation.unbiased_report?.certification_status}</span>
              </div>
              <div className="p-4 rounded-xl bg-emerald-950/50 border border-emerald-500/20">
                <span className="text-xs text-emerald-500 font-bold uppercase tracking-wider block mb-1">Final Metrics</span>
                <span className="text-lg text-emerald-100">{simulation.unbiased_report?.final_metrics}</span>
              </div>
              <div className="p-4 rounded-xl bg-emerald-950/50 border border-emerald-500/20">
                <span className="text-xs text-emerald-500 font-bold uppercase tracking-wider block mb-1">Recommendation</span>
                <span className="text-lg text-emerald-100">{simulation.unbiased_report?.release_recommendation}</span>
              </div>
            </div>
            {csv_string && (
              <button onClick={handleDownload} className="w-full flex items-center justify-center gap-3 bg-emerald-500 hover:bg-emerald-400 text-emerald-950 font-bold py-4 rounded-2xl transition-all shadow-[0_0_30px_rgba(16,185,129,0.4)] hover:shadow-[0_0_40px_rgba(16,185,129,0.6)]">
                <FileText className="w-6 h-6" /> Download Certified Unbiased Dataset (CSV)
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MethodSelectorModal({ onSelect, onCancel }) {
  const [selected, setSelected] = useState('exponentiated_gradient');
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="glass-dark rounded-3xl p-8 max-w-2xl w-full border border-emerald-500/30">
        <h3 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
          <Zap className="w-6 h-6 text-emerald-400" /> Choose Mitigation Algorithm
        </h3>
        <p className="text-gray-400 text-sm mb-6">Each algorithm tackles bias at a different stage of the ML pipeline.</p>
        <div className="space-y-3 mb-8">
          {MITIGATION_METHODS.map((m) => (
            <button key={m.key} onClick={() => setSelected(m.key)}
              className={`w-full text-left p-4 rounded-2xl border transition-all ${selected === m.key ? 'border-emerald-500/60 bg-emerald-900/20' : 'border-gray-700/50 bg-gray-800/30 hover:border-gray-600'}`}>
              <div className="flex items-center gap-3 mb-1">
                <span className="text-xl">{m.icon}</span>
                <span className="font-semibold text-white">{m.label}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full border ${m.badgeColor}`}>{m.badge}</span>
                {selected === m.key && <CheckCircle className="w-4 h-4 text-emerald-400 ml-auto" />}
              </div>
              <p className="text-gray-400 text-sm pl-9">{m.description}</p>
            </button>
          ))}
        </div>
        <div className="flex gap-3">
          <button onClick={() => onSelect(selected)} className="flex-1 flex items-center justify-center gap-2 bg-emerald-500 hover:bg-emerald-400 text-slate-900 font-bold py-3 px-8 rounded-xl transition-all">
            <Zap className="w-5 h-5" /> Run Mitigation
          </button>
          <button onClick={onCancel} className="px-6 py-3 rounded-xl bg-gray-700/50 hover:bg-gray-700 text-gray-300 transition">Cancel</button>
        </div>
      </div>
    </div>
  );
}

// ── BUG 4 FIX: Column picker step ────────────────────────────────────────────
function ColumnPicker({ columns, sampleRows, suggestedTarget, suggestedSensitive, totalRows, filename, onConfirm, onCancel }) {
  const [targetCol,    setTargetCol]    = useState(suggestedTarget    || columns[columns.length - 1] || '');
  const [sensitiveCol, setSensitiveCol] = useState(suggestedSensitive || columns[0] || '');

  const conflict = targetCol === sensitiveCol;

  return (
    <div className="glass-dark rounded-3xl p-8 max-w-3xl mx-auto border border-cyan-500/30 animate-fade-in">
      <div className="flex items-center gap-3 mb-2">
        <Columns className="w-6 h-6 text-cyan-400" />
        <h2 className="text-xl font-bold text-white">Select Columns</h2>
      </div>
      <p className="text-gray-400 text-sm mb-6">
        <span className="text-cyan-300 font-medium">{filename}</span> — {totalRows.toLocaleString()} rows, {columns.length} columns detected.
        Choose which column is the <strong className="text-white">outcome</strong> (what the model predicts) and which is the <strong className="text-white">protected attribute</strong> (what to check for bias).
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-6">
        <div>
          <label className="block text-sm font-semibold text-gray-400 mb-2 uppercase tracking-wider">
            Target column <span className="text-rose-400">*</span>
          </label>
          <select
            value={targetCol}
            onChange={(e) => setTargetCol(e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-cyan-500 transition"
          >
            {columns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <p className="text-xs text-gray-500 mt-1">The binary outcome the model predicts (e.g. hired, approved)</p>
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-400 mb-2 uppercase tracking-wider">
            Sensitive attribute <span className="text-rose-400">*</span>
          </label>
          <select
            value={sensitiveCol}
            onChange={(e) => setSensitiveCol(e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-cyan-500 transition"
          >
            {columns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <p className="text-xs text-gray-500 mt-1">Protected attribute to audit for bias (e.g. gender, race)</p>
        </div>
      </div>

      {conflict && (
        <div className="flex items-center gap-2 text-rose-300 bg-rose-500/10 border border-rose-500/20 p-3 rounded-xl mb-4 text-sm">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          Target and sensitive columns must be different.
        </div>
      )}

      {/* Sample data preview */}
      {sampleRows.length > 0 && (
        <div className="mb-6 overflow-x-auto rounded-xl border border-gray-700/50">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-gray-800/80">
                {columns.map((c) => (
                  <th key={c} className={`px-3 py-2 text-left font-semibold whitespace-nowrap ${c === targetCol ? 'text-rose-300' : c === sensitiveCol ? 'text-cyan-300' : 'text-gray-400'}`}>
                    {c === targetCol ? '🎯 ' : c === sensitiveCol ? '🔍 ' : ''}{c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sampleRows.map((row, i) => (
                <tr key={i} className="border-t border-gray-700/30 hover:bg-gray-800/30">
                  {columns.map((c) => <td key={c} className="px-3 py-2 text-gray-300">{row[c]}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="px-3 py-2 text-xs text-gray-600 bg-gray-900/30">Showing first 3 rows. 🎯 = target column &nbsp;·&nbsp; 🔍 = sensitive column</p>
        </div>
      )}

      <div className="flex gap-3">
        <button
          onClick={() => onConfirm(targetCol, sensitiveCol)}
          disabled={conflict}
          className="flex-1 flex items-center justify-center gap-2 bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed text-slate-900 font-bold py-3 px-8 rounded-xl transition-all"
        >
          <CheckCircle className="w-5 h-5" /> Run Bias Audit
        </button>
        <button onClick={onCancel} className="px-6 py-3 rounded-xl bg-gray-700/50 hover:bg-gray-700 text-gray-300 transition">
          Back
        </button>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  // dataState.state:  IDLE | COLUMN_PICKING | LOADING | RESOLVED | ERROR
  const [dataState,     setDataState]     = useState({ state: 'IDLE', data: null, error: null });
  const [mitState,      setMitState]      = useState({ state: 'IDLE', data: null, error: null });
  const [showModal,     setShowModal]     = useState(false);
  const [activeDataset, setActiveDataset] = useState(null);
  const [columnInfo,    setColumnInfo]    = useState(null); // { columns, sampleRows, suggestedTarget, suggestedSensitive, totalRows, file }

  const datasetLinks = [
    { name: 'Corporate Hiring',      id: 'corporate_hiring' },
    { name: 'Mortgage Approvals',    id: 'mortgage_approvals' },
    { name: 'Hospital Triage',       id: 'hospital_triage' },
    { name: 'Criminal Recidivism',   id: 'criminal_recidivism' },
    { name: 'University Admissions', id: 'university_admissions' },
  ];

  const handleReset = () => {
    setDataState({ state: 'IDLE', data: null, error: null });
    setMitState({ state: 'IDLE', data: null, error: null });
    setActiveDataset(null);
    setColumnInfo(null);
  };

  // ── Step 1: fetch column list from uploaded file ────────────────────────────
  const onFileSelect = async (e) => {
    e.preventDefault();
    const file = e.dataTransfer ? e.dataTransfer.files[0] : e.target.files[0];
    if (!file) return;

    setDataState({ state: 'LOADING', data: null, error: null });

    const formData = new FormData();
    formData.append('file', file);

    try {
      // BUG 1 FIX: NO headers block — let browser set Content-Type + boundary
      const res = await axios.post(`${API_BASE}/columns`, formData);
      setColumnInfo({
        columns:           res.data.columns,
        sampleRows:        res.data.sample_rows,
        suggestedTarget:   res.data.suggested_target_col,
        suggestedSensitive:res.data.suggested_sensitive_col,
        totalRows:         res.data.total_rows,
        file,
      });
      setDataState({ state: 'COLUMN_PICKING', data: null, error: null });
    } catch (err) {
      // BUG 2 FIX: extract actual FastAPI detail
      setDataState({ state: 'ERROR', data: null, error: extractError(err) });
    }
  };

  // ── Step 2: user confirmed columns → run full audit ─────────────────────────
  const runAudit = async (targetCol, sensitiveCol) => {
    setDataState({ state: 'LOADING', data: null, error: null });
    setMitState({ state: 'IDLE', data: null, error: null });
    setActiveDataset({ type: 'upload', file: columnInfo.file, targetCol, sensitiveCol });

    const formData = new FormData();
    formData.append('file', columnInfo.file);

    try {
      // BUG 1 FIX: NO headers block
      // BUG 4 FIX: column names sent as query params
      const res = await axios.post(
        `${API_BASE}/audit?target_col=${encodeURIComponent(targetCol)}&sensitive_col=${encodeURIComponent(sensitiveCol)}`,
        formData
        // ← no headers object here
      );
      setDataState({ state: 'RESOLVED', data: res.data, error: null });
    } catch (err) {
      // BUG 2 FIX
      setDataState({ state: 'ERROR', data: null, error: extractError(err) });
    }
  };

  // ── Load preset dataset ─────────────────────────────────────────────────────
  const loadTestDataset = async (datasetId) => {
    setDataState({ state: 'LOADING', data: null, error: null });
    setMitState({ state: 'IDLE', data: null, error: null });
    setActiveDataset({ type: 'preset', id: datasetId });
    try {
      const res = await axios.get(`${API_BASE}/test_audit/${datasetId}`);
      setDataState({ state: 'RESOLVED', data: res.data, error: null });
    } catch (err) {
      // BUG 2 FIX
      setDataState({ state: 'ERROR', data: null, error: extractError(err) });
    }
  };

  // ── Apply mitigation ────────────────────────────────────────────────────────
  const applyMitigation = async () => {
    setShowModal(false);
    setMitState({ state: 'LOADING', data: null, error: null });

    try {
      let res;
      if (activeDataset?.type === 'preset') {
        res = await axios.get(`${API_BASE}/mitigate/${activeDataset.id}`);
      } else if (activeDataset?.type === 'upload') {
        const { file, targetCol, sensitiveCol } = activeDataset;
        const formData = new FormData();
        formData.append('file', file);
        res = await axios.post(
          `${API_BASE}/mitigate?target_col=${encodeURIComponent(targetCol)}&sensitive_col=${encodeURIComponent(sensitiveCol)}`,
          formData
        );
      } else {
        throw new Error('No active dataset. Please run an audit first.');
      }
      setMitState({ state: 'RESOLVED', data: res.data, error: null });
    } catch (err) {
      setMitState({ state: 'ERROR', data: null, error: extractError(err) });
    }
  };

  const handleDragOver = (e) => e.preventDefault();
  const handleDrop     = (e) => onFileSelect(e);

  const getAgentColor = (a) => {
    if (a.includes('Data Scientist')) return 'text-blue-400 bg-blue-400/10 border-blue-400/30';
    if (a.includes('Ethics'))         return 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30';
    if (a.includes('Legal') || a.includes('Compliance')) return 'text-rose-400 bg-rose-400/10 border-rose-400/30';
    return 'text-purple-400 bg-purple-400/10 border-purple-400/30';
  };
  const getAgentIcon = (a) => {
    if (a.includes('Data Scientist')) return <FileText   className="w-5 h-5 text-blue-400" />;
    if (a.includes('Ethics'))         return <ShieldCheck className="w-5 h-5 text-emerald-400" />;
    if (a.includes('Legal') || a.includes('Compliance')) return <Scale className="w-5 h-5 text-rose-400" />;
    return <AlertTriangle className="w-5 h-5 text-purple-400" />;
  };

  // ── Mitigation result view ──────────────────────────────────────────────────
  if (mitState.state === 'RESOLVED' && mitState.data) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <Header />
        <MitigationResultPanel result={mitState.data} onReset={handleReset} />
        <Styles />
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <Header />

      {/* ── IDLE ─────────────────────────────────────────────────────────── */}
      {dataState.state === 'IDLE' && (
        <div className="space-y-8 animate-fade-in text-center">
          <div onDragOver={handleDragOver} onDrop={handleDrop}
            className="border-2 border-dashed border-gray-600 rounded-3xl p-20 glass-dark transition-all hover:border-cyan-500/50 hover:bg-gray-800/80 cursor-pointer group">
            <input type="file" id="fileUpload" className="hidden" onChange={onFileSelect} accept=".csv" />
            <label htmlFor="fileUpload" className="cursor-pointer flex flex-col items-center">
              <UploadCloud className="w-20 h-20 text-gray-500 mb-6 group-hover:text-cyan-400 transition-colors duration-300" />
              <h3 className="text-2xl font-semibold mb-2">Drag & drop your dataset here</h3>
              <p className="text-gray-400">or click to browse (.csv format)</p>
            </label>
          </div>
          <div className="glass-dark p-8 rounded-3xl border border-gray-700/50">
            <h3 className="text-xl font-medium mb-6 text-gray-300">Run test datasets instantly:</h3>
            <div className="flex flex-wrap justify-center gap-4">
              {datasetLinks.map((link) => (
                <button key={link.id} onClick={() => loadTestDataset(link.id)}
                  className="px-6 py-3 rounded-full bg-gray-800/50 border border-gray-700 hover:border-cyan-500/50 hover:bg-cyan-950/30 transition-all text-sm font-medium">
                  {link.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── COLUMN PICKER (Bug 4 fix: new step) ──────────────────────────── */}
      {dataState.state === 'COLUMN_PICKING' && columnInfo && (
        <ColumnPicker
          columns={columnInfo.columns}
          sampleRows={columnInfo.sampleRows}
          suggestedTarget={columnInfo.suggestedTarget}
          suggestedSensitive={columnInfo.suggestedSensitive}
          totalRows={columnInfo.totalRows}
          filename={columnInfo.file.name}
          onConfirm={runAudit}
          onCancel={handleReset}
        />
      )}

      {/* ── LOADING ───────────────────────────────────────────────────────── */}
      {dataState.state === 'LOADING' && (
        <div className="flex flex-col items-center justify-center py-32 space-y-6">
          <div className="relative w-24 h-24">
            <div className="absolute inset-0 rounded-full border-t-4 border-cyan-400 animate-spin" />
            <div className="absolute inset-2 rounded-full border-r-4 border-purple-500 animate-spin animation-delay-150" />
            <div className="absolute inset-4 rounded-full border-b-4 border-rose-400 animate-spin animation-delay-300" />
          </div>
          <h3 className="text-xl font-medium text-cyan-100 animate-pulse">Running MIROFISH Analysis & Board Debate...</h3>
        </div>
      )}

      {/* ── ERROR ────────────────────────────────────────────────────────── */}
      {dataState.state === 'ERROR' && (
        <div className="glass-dark border-red-500/30 p-8 rounded-3xl text-center max-w-2xl mx-auto">
          <AlertTriangle className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <h3 className="text-2xl font-bold text-red-200 mb-2">Analysis Failed</h3>
          {/* BUG 2 FIX: shows the real FastAPI detail string now */}
          <p className="text-red-400/80 font-mono text-sm bg-red-900/20 p-3 rounded-lg">{dataState.error}</p>
          <button onClick={handleReset} className="mt-6 px-6 py-2 bg-red-500/20 text-red-300 rounded-lg hover:bg-red-500/30 transition-colors">
            Try Again
          </button>
        </div>
      )}

      {/* ── RESOLVED ─────────────────────────────────────────────────────── */}
      {dataState.state === 'RESOLVED' && dataState.data && (
        <div className="space-y-6 animate-fade-in">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-cyan-50">Analysis Results</h2>
            <button onClick={handleReset} className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-gray-800/80 hover:bg-gray-700 transition">
              <RefreshCw className="w-4 h-4" /> <span>New Audit</span>
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass-dark rounded-3xl p-6 flex flex-col">
              <div className="mb-6 flex items-center justify-between border-b border-gray-700/50 pb-4">
                <h3 className="text-xl font-semibold flex items-center gap-2">
                  <BarChart className="w-5 h-5 text-cyan-400" /> Model Fairness Metrics
                </h3>
                <span className="text-sm px-3 py-1 bg-gray-800 rounded-full text-gray-400 border border-gray-700">
                  Accuracy: {(dataState.data.metrics.model_metrics.accuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex-1 min-h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={[{ name: 'Demographic Parity Gap', value: dataState.data.metrics.bias_metrics.demographic_parity_difference * 100 }]} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                    <XAxis dataKey="name" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" tickFormatter={(v) => `${v}%`} />
                    <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ backgroundColor: 'rgba(17,24,39,0.9)', borderColor: '#374151', borderRadius: '8px' }} />
                    <Legend />
                    <Bar dataKey="value" name="Bias % Gap" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 p-4 bg-purple-500/10 border border-purple-500/20 rounded-xl">
                <p className="text-sm text-purple-200">
                  <strong className="text-purple-400">Insight:</strong> The model exhibits a{' '}
                  <span className="font-bold mx-1 text-white">{(dataState.data.metrics.bias_metrics.demographic_parity_difference * 100).toFixed(2)}%</span>
                  difference in positive prediction rates between demographic groups.
                </p>
              </div>
            </div>

            <div className="glass-dark rounded-3xl p-6 flex flex-col max-h-[600px]">
              <h3 className="text-xl font-semibold mb-4 border-b border-gray-700/50 pb-4 flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-emerald-400" /> MiroFish Council Debate
              </h3>
              <div className="flex-1 overflow-y-auto pr-2 space-y-4 custom-scrollbar">
                {dataState.data.simulation?.debate?.map((turn, i) => (
                  <div key={i} className={`flex flex-col p-4 rounded-2xl border ${getAgentColor(turn.agent)}`}>
                    <div className="flex items-center gap-2 mb-2">
                      {getAgentIcon(turn.agent)}
                      <span className="font-bold text-sm tracking-wide uppercase">{turn.agent}</span>
                    </div>
                    <p className="text-gray-200 text-sm leading-relaxed">{turn.dialogue}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="glass-dark rounded-3xl p-8 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none" />
            <h3 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <FileText className="w-6 h-6 text-cyan-400" /> Official Simulation Report
            </h3>
            <div className="grid md:grid-cols-2 gap-8 mb-8">
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Identified Harm</h4>
                <p className="text-lg text-rose-200 bg-rose-500/5 p-4 rounded-xl border border-rose-500/20">{dataState.data.simulation?.report?.identified_harm}</p>
              </div>
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">PR & Legal Forecast</h4>
                <p className="text-lg text-yellow-200 bg-yellow-500/5 p-4 rounded-xl border border-yellow-500/20">{dataState.data.simulation?.report?.pr_and_legal_forecast}</p>
              </div>
            </div>
            <div className="bg-gradient-to-r from-emerald-900/40 to-cyan-900/40 border border-emerald-500/30 p-6 rounded-2xl">
              <h4 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-2">Proposed Mitigation</h4>
              <p className="text-emerald-50 text-lg mb-6">{dataState.data.simulation?.report?.proposed_mitigation}</p>
              {mitState.state === 'IDLE' && (
                <button onClick={() => applyMitigation()}
                  className="flex items-center justify-center w-full sm:w-auto gap-2 bg-emerald-500 hover:bg-emerald-400 text-slate-900 font-bold py-3 px-8 rounded-xl transition-all shadow-[0_0_20px_rgba(16,185,129,0.3)]">
                  <CheckCircle className="w-5 h-5" /> Apply Recommended Mitigation
                </button>
              )}
              {mitState.state === 'LOADING' && (
                <div className="flex items-center gap-3 text-emerald-300">
                  <div className="w-5 h-5 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
                  <span className="font-semibold">Calculating AIF360 Weights & Re-Simulating Council...</span>
                </div>
              )}
              {mitState.state === 'ERROR' && (
                <div className="flex items-center gap-3 text-rose-300 bg-rose-500/10 border border-rose-500/20 p-4 rounded-xl">
                  <AlertTriangle className="w-5 h-5 shrink-0" />
                  {/* BUG 2 FIX: real error shown here too */}
                  <span className="font-mono text-sm">{mitState.error}</span>
                  <button onClick={() => setMitState({ state: 'IDLE', data: null, error: null })} className="ml-auto text-xs underline">Retry</button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      <Styles />
    </div>
  );
}

function Header() {
  return (
    <header className="mb-10 text-center">
      <h1 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-500 mb-4">
        AI Panchayat Dashboard
      </h1>
      <p className="text-gray-400 text-lg">Autonomous AI Bias Auditing & Mitigation Simulator</p>
    </header>
  );
}

function Styles() {
  return (
    <style dangerouslySetInnerHTML={{ __html: `
      .custom-scrollbar::-webkit-scrollbar { width: 6px; }
      .custom-scrollbar::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); border-radius: 10px; }
      .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
      .animation-delay-150 { animation-delay: 150ms; }
      .animation-delay-300 { animation-delay: 300ms; }
      @keyframes fade-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
      .animate-fade-in { animation: fade-in 0.4s ease both; }
    `}} />
  );
}
