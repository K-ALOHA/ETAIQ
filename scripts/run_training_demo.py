"""Run a quick training demo using the ETAIQ training service.

This script loads the engineered dataset, runs the training service,
persists the best model, and performs a sample prediction to validate
the end-to-end pipeline.
"""

from pathlib import Path
import traceback

from ml.features.data_split import DataSplitEngine
from ml.training.training_service import TrainingService
from ml.training.prediction_pipeline import PredictionPipelineEngine


def main() -> None:
    try:
        splitter = DataSplitEngine()
        df = splitter.load_dataset()
        X_train, X_test, y_train, y_test = splitter.split_dataset(df)

        print("Running training service...")
        service = TrainingService()
        result = service.train(X_train, X_test, y_train, y_test)

        print("Training completed")
        print(f"Best model: {result.best_model.model_name} v{result.saved_model.version}")
        print(f"Saved model path: {result.saved_model.model_path}")

        # Run a quick prediction using the persisted model
        sample_row = X_test.iloc[0:1]
        pipeline = PredictionPipelineEngine()
        pred_result = pipeline.predict(result.saved_model.model_path, sample_row)
        print("Sample prediction:", pred_result.predictions[0])

    except Exception as exc:
        print("Demo failed:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
