import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import { UploadCloud, FileText, AlertTriangle, ShieldCheck, Scale, CheckCircle, RefreshCw } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

export default function App() {
  const [dataState, setDataState] = useState({ state: 'IDLE', data: null, error: null });

  const datasetLinks = [
    { name: 'Corporate Hiring', id: 'corporate_hiring' },
    { name: 'Mortgage Approvals', id: 'mortgage_approvals' },
    { name: 'Hospital Triage', id: 'hospital_triage' },
    { name: 'Criminal Recidivism', id: 'criminal_recidivism' },
    { name: 'University Admissions', id: 'university_admissions' },
  ];

  const loadTestDataset = async (datasetId) => {
    setDataState({ state: 'LOADING', data: null, error: null });
    try {
      const response = await axios.get(`${API_BASE}/test_audit/${datasetId}`);
      setDataState({ state: 'RESOLVED', data: response.data, error: null });
    } catch (err) {
      setDataState({ state: 'ERROR', data: null, error: err.message || 'Failed to load dataset' });
    }
  };

  const onFileUpload = async (e) => {
    e.preventDefault();
    const file = e.dataTransfer ? e.dataTransfer.files[0] : e.target.files[0];
    if (!file) return;

    setDataState({ state: 'LOADING', data: null, error: null });
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post(`${API_BASE}/audit`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setDataState({ state: 'RESOLVED', data: response.data, error: null });
    } catch (err) {
      setDataState({ state: 'ERROR', data: null, error: err.message || 'Failed to analyze file' });
    }
  };

  const handleDragOver = (e) => e.preventDefault();
  const handleDrop = (e) => onFileUpload(e);

  const getAgentColor = (agentName) => {
    if (agentName.includes('Data Scientist')) return 'text-blue-400 bg-blue-400/10 border-blue-400/30';
    if (agentName.includes('Ethics')) return 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30';
    if (agentName.includes('Legal') || agentName.includes('Compliance')) return 'text-rose-400 bg-rose-400/10 border-rose-400/30';
    return 'text-purple-400 bg-purple-400/10 border-purple-400/30';
  };

  const getAgentIcon = (agentName) => {
    if (agentName.includes('Data Scientist')) return <FileText className="w-5 h-5 text-blue-400" />;
    if (agentName.includes('Ethics')) return <ShieldCheck className="w-5 h-5 text-emerald-400" />;
    if (agentName.includes('Legal') || agentName.includes('Compliance')) return <Scale className="w-5 h-5 text-rose-400" />;
    return <AlertTriangle className="w-5 h-5 text-purple-400" />;
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <header className="mb-10 text-center">
        <h1 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-500 mb-4">
          AI Panchayat Dashboard
        </h1>
        <p className="text-gray-400 text-lg">Autonomous AI Bias Auditing & Mitigation Simulator</p>
      </header>

      {dataState.state === 'IDLE' && (
        <div className="space-y-8 animate-fade-in text-center">
          <div 
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className="border-2 border-dashed border-gray-600 rounded-3xl p-20 glass-dark transition-all hover:border-cyan-500/50 hover:bg-gray-800/80 cursor-pointer group"
          >
            <input type="file" id="fileUpload" className="hidden" onChange={onFileUpload} accept=".csv" />
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
                <button
                  key={link.id}
                  onClick={() => loadTestDataset(link.id)}
                  className="px-6 py-3 rounded-full bg-gray-800/50 border border-gray-700 hover:border-cyan-500/50 hover:bg-cyan-950/30 transition-all text-sm font-medium hover:shadow-[0_0_15px_rgba(34,211,238,0.2)]"
                >
                  {link.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {dataState.state === 'LOADING' && (
        <div className="flex flex-col items-center justify-center py-32 space-y-6">
          <div className="relative w-24 h-24">
            <div className="absolute inset-0 rounded-full border-t-4 border-cyan-400 animate-spin"></div>
            <div className="absolute inset-2 rounded-full border-r-4 border-purple-500 animate-spin animation-delay-150"></div>
            <div className="absolute inset-4 rounded-full border-b-4 border-rose-400 animate-spin animation-delay-300"></div>
          </div>
          <h3 className="text-xl font-medium text-cyan-100 animate-pulse">Running MIROFISH Analysis & Board Debate...</h3>
        </div>
      )}

      {dataState.state === 'ERROR' && (
        <div className="glass-dark border-red-500/30 p-8 rounded-3xl text-center max-w-2xl mx-auto">
          <AlertTriangle className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <h3 className="text-2xl font-bold text-red-200 mb-2">Analysis Failed</h3>
          <p className="text-red-400/80">{dataState.error}</p>
          <button 
            onClick={() => setDataState({ state: 'IDLE', data: null, error: null })}
            className="mt-6 px-6 py-2 bg-red-500/20 text-red-300 rounded-lg hover:bg-red-500/30 transition-colors"
          >
            Try Again
          </button>
        </div>
      )}

      {dataState.state === 'RESOLVED' && dataState.data && (
        <div className="space-y-6 animate-fade-in">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-cyan-50">Analysis Results</h2>
            <button 
              onClick={() => setDataState({ state: 'IDLE', data: null, error: null })}
              className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-gray-800/80 hover:bg-gray-700 transition"
            >
              <RefreshCw className="w-4 h-4" /> <span>New Audit</span>
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Column: Mathematical Metrics */}
            <div className="glass-dark rounded-3xl p-6 flex flex-col">
              <div className="mb-6 flex items-center justify-between border-b border-gray-700/50 pb-4">
                <h3 className="text-xl font-semibold flex items-center gap-2">
                  <BarChart className="w-5 h-5 text-cyan-400" /> Model Fairness Metrics
                </h3>
                <span className="text-sm px-3 py-1 bg-gray-800 rounded-full text-gray-400 border border-gray-700">
                  Base Accuracy: {(dataState.data.metrics.model_metrics.accuracy * 100).toFixed(1)}%
                </span>
              </div>
              
              <div className="flex-1 min-h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={[
                      { name: 'Demographic Parity Gap', value: dataState.data.metrics.bias_metrics.demographic_parity_difference * 100, fill: '#8b5cf6' }
                    ]}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                    <XAxis dataKey="name" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" tickFormatter={(v) => `${v}%`} />
                    <Tooltip 
                      cursor={{fill: 'rgba(255,255,255,0.05)'}}
                      contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.9)', borderColor: '#374151', borderRadius: '8px' }} 
                    />
                    <Legend />
                    <Bar dataKey="value" name="Bias Percentage Gap" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 p-4 bg-purple-500/10 border border-purple-500/20 rounded-xl">
                <p className="text-sm text-purple-200">
                  <strong className="text-purple-400">Insight:</strong> The model exhibits a 
                  <span className="font-bold mx-1 text-white">
                    {(dataState.data.metrics.bias_metrics.demographic_parity_difference * 100).toFixed(2)}% 
                  </span> 
                  difference in positive prediction rates between demographic groups.
                </p>
              </div>
            </div>

            {/* Right Column: MiroFish Council Debate */}
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

          {/* Bottom Span: Official Simulation Report */}
          <div className="glass-dark rounded-3xl p-8 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none"></div>
            
            <h3 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <FileText className="w-6 h-6 text-cyan-400" /> Official Simulation Report
            </h3>
            
            <div className="grid md:grid-cols-2 gap-8 mb-8">
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Identified Harm</h4>
                <p className="text-lg text-rose-200 bg-rose-500/5 p-4 rounded-xl border border-rose-500/20">
                  {dataState.data.simulation?.report?.identified_harm}
                </p>
              </div>
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">PR & Legal Forecast</h4>
                <p className="text-lg text-yellow-200 bg-yellow-500/5 p-4 rounded-xl border border-yellow-500/20">
                  {dataState.data.simulation?.report?.pr_and_legal_forecast}
                </p>
              </div>
            </div>

            <div className="bg-gradient-to-r from-emerald-900/40 to-cyan-900/40 border border-emerald-500/30 p-6 rounded-2xl">
              <h4 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-2">Proposed Mitigation</h4>
              <p className="text-emerald-50 text-lg mb-6">
                {dataState.data.simulation?.report?.proposed_mitigation}
              </p>
              <button className="flex items-center justify-center w-full sm:w-auto gap-2 bg-emerald-500 hover:bg-emerald-400 text-slate-900 font-bold py-3 px-8 rounded-xl transition-all shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:shadow-[0_0_30px_rgba(16,185,129,0.5)]">
                <CheckCircle className="w-5 h-5" /> Apply Recommended Mitigation
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Global minimal scrollbar and utilities injected via style if necessary */}
      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
        .animation-delay-150 { animation-delay: 150ms; }
        .animation-delay-300 { animation-delay: 300ms; }
      `}} />
    </div>
  );
}
