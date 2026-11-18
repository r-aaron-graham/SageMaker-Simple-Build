# SageMaker Scikit-learn Training Example

Train a linear regression model on reproducible synthetic data and save a Joblib artifact. The training script accepts a local output path and also supports SageMaker's `SM_MODEL_DIR` environment variable.

## Run locally

From the repository root, using Python 3 in an activated virtual environment:

```sh
python -m pip install -r requirements.txt
python "SageMaker SKLearn estimator.py" --model-dir ./local-model
python -c "import joblib; model = joblib.load('local-model/model.joblib'); print(model.predict([[0.5, 0.2, 0.8]]))"
```

Training writes `local-model/model.joblib`. The final command loads the artifact you just produced and predicts a value close to 3.25. AWS credentials are unnecessary for this local run.

## Source and deployment status

- [Training script](SageMaker%20SKLearn%20estimator.py) generates 100 synthetic rows, fits a scikit-learn linear regression, and writes the resulting model.
- [Endpoint deployment sketch](SageMaker%20Python%20SDK%20to%20deploy%20the%20trained%20model%20as%20an%20endpoint.py) illustrates how a model would be deployed with the SageMaker SDK. It references a placeholder S3 artifact, an older framework version, and the included JSON inference entry point. Confirm the framework and instance availability for the target region and plan endpoint cleanup before adapting it.

## Prediction interface

The included inference.py implements loading, validation, prediction, and JSON serialization. Use application/json with a nonempty array of three-feature rows, such as [[0.5, 0.2, 0.8]]. Invalid shapes and nonfinite values raise a validation error. Load Joblib artifacts only from a trusted training process.
