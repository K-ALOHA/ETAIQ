"""Run a quick sampled training smoke test for ETAIQ.

This performs a small-sample training run to validate the training pipeline,
model persistence, registry registration, and a sample prediction without
processing the entire production dataset.
"""

from pathlib import Path
import traceback

import pandas as pd

from ml.features.data_split import DataSplitEngine
from ml.training.training_service import TrainingService
from ml.training.prediction_pipeline import PredictionPipelineEngine


def main(sample_size: int = 2000) -> None:
    try:
        splitter = DataSplitEngine()
        df = splitter.load_dataset()

        # Sample a subset for a quick smoke test
        sample = df.sample(n=min(sample_size, len(df)), random_state=42)
        X_train, X_test, y_train, y_test = splitter.split_dataset(sample)

        print("Running sampled training service... (rows=", len(sample), ")")
        service = TrainingService()
        result = service.train(X_train, X_test, y_train, y_test)

        print("Training completed")
        print(f"Best model: {result.best_model.model_name} v{result.saved_model.version}")
        print(f"Saved model path: {result.saved_model.model_path}")

        # Run a quick prediction using the persisted model
        sample_row = X_test.iloc[0:1]
        pipeline = PredictionPipelineEngine()
        pred_result = pipeline.predict(result.saved_model.model_path, sample_row)
        print("Sample prediction:", float(pred_result.predictions[0]))

    except Exception as exc:
        print("Smoke demo failed:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
