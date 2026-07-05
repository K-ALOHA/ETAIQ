#!/usr/bin/env python3
import sys
from pathlib import Path
import traceback
import asyncio

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
from pathlib import Path

# Simulate get_dataset()
def simulate_get_dataset():
    repo_root = Path(__file__).resolve()
    dataset_path = repo_root / "ml" / "data" / "features" / "engineered_training_dataset.csv"
    print(f"Dataset path: {dataset_path}")
    print(f"Exists? {dataset_path.exists()}")
    
    dataframe = pd.read_csv(dataset_path)
    missing_summary = dataframe.isna().sum().to_dict()
    missing_values = {k: int(v) for k, v in missing_summary.items() if v > 0}
    
    return {
        "record_count": int(len(dataframe)),
        "feature_count": len(dataframe.columns),
        "target_column": "actual_delivery_time_min",
        "missing_values": missing_values,
        "feature_names": [str(col) for col in dataframe.columns],
    }

# Simulate _handle_dataset_summary
def simulate_handle_dataset_summary():
    try:
        # Import get_dataset directly (since asyncio.run is in the original)
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "backend"))
        
        import sys
        from pathlib import Path
        # Add minimal things
        sys.path.insert(0, str(Path(__file__).parent))
        sys.path.insert(0, str(Path(__file__).parent / "backend"))
        
        # Let's manually import without touching app imports that load settings! Let's copy the get_dataset() function!
        from pathlib import Path
        import pandas as pd
        
        def manual_get_dataset():
            repo_root = Path(__file__).resolve()  # Same as get_dataset does!
            dataset_path = repo_root / "ml" / "data" / "features" / "engineered_training_dataset.csv"
            print(f"Manual dataset path: {dataset_path}")
            if not dataset_path.exists():
                print("Not exist!")
                return {
                    "record_count": 0,
                    "feature_count":0,
                    "target_column": None,
                    "missing_values": {},
                    "feature_names": []
                }
            dataframe = pd.read_csv(dataset_path)
            missing_summary = dataframe.isna().sum().to_dict()
            missing_values = {k: int(v) for k, v in missing_summary.items() if v >0}
            return {
                "record_count": len(dataframe),
                "feature_count": len(dataframe.columns),
                "target_column": "actual_delivery_time_min",
                "missing_values": missing_values,
                "feature_names": [str(col) for col in dataframe.columns]
            }
        
        # Call it!
        dataset = manual_get_dataset()
        
        # Format like assistant!
        return (
            "Dataset summary:\n"
            f"  Number of records: {dataset['record_count']}\n"
            f"  Number of features: {dataset['feature_count']}\n"
            f"  Target column: {dataset['target_column']}\n"
            f"  Missing values: {', '.join([f'{k} ({v})' for k, v in dataset['missing_values'].items()]) or 'None'}"
        )
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        print(traceback.format_exc())

# Run
result = simulate_handle_dataset_summary()
print(f"\nResult: {result}")
