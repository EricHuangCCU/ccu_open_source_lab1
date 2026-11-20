"""Lab 1 - Step 2: train an Iris classifier and track the run with MLflow."""
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

N_ESTIMATORS = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

mlflow.set_experiment("iris")
with mlflow.start_run() as run:
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    clf.fit(Xtr, ytr)
    accuracy = clf.score(Xte, yte)

    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")

    joblib.dump(clf, "model.pkl")
    print(f"run_id={run.info.run_id} accuracy={accuracy:.4f} -> model.pkl saved")

# View runs:  mlflow ui  ->  http://localhost:5000
