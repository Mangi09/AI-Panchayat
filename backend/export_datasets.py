import os
import sys

# Ensure app is in path if run from backend dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.data_generator import AVAILABLE_DATASETS, get_test_dataset

def export_all():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    os.makedirs(out_dir, exist_ok=True)
    
    for name in AVAILABLE_DATASETS:
        info = get_test_dataset(name)
        df = info['dataframe']
        # Also let's save a metadata text file so they know the target and sensitive col?
        # Nah, just the CSV. 
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"Successfully exported: {path}")

if __name__ == '__main__':
    export_all()
