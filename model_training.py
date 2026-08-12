from pathlib import Path
from urllib.request import urlopen

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/neeravjain91-jpg/forest-fire-prediction-/main/Algerian_forest_fires_cleaned_dataset.csv"
MODEL_PATH = Path("models/fire_classifier.joblib")
METRICS_PATH = Path("models/metrics.json")

FEATURES = [
    "Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "Region"
]


def load_data():
    """Load the cleaned Algerian Forest Fires dataset from the original project."""
    local_path = Path("Algerian_forest_fires_cleaned_dataset.csv")
    if local_path.exists():
        df = pd.read_csv(local_path)
    else:
        with urlopen(DATA_URL, timeout=30) as response:
            df = pd.read_csv(response)

    df.columns = df.columns.str.strip()
    df["Classes"] = df["Classes"].astype(str).str.strip().str.lower()
    df["target"] = df["Classes"].map({"not fire": 0, "fire": 1})
    df = df.dropna(subset=FEATURES + ["target"])
    return df


def train_models():
    df = load_data()
    X = df[FEATURES]
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted = {}

    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results[name] = {
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
            "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
            "f1": round(f1_score(y_test, predictions, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        }
        fitted[name] = estimator

    best_name = max(results, key=lambda name: results[name]["f1"])
    best_model = fitted[best_name]

    # Refit the selected model on the complete dataset before deployment.
    best_model.fit(X, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    metrics = {
        "selected_model": best_name,
        "features": FEATURES,
        "test_size": 0.20,
        "random_state": 42,
        "models": results,
    }

    import json
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return best_model, metrics


def load_or_train_model():
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        model = joblib.load(MODEL_PATH)
        import json
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return model, metrics
    return train_models()


if __name__ == "__main__":
    model, metrics = train_models()
    print(f"Selected model: {metrics['selected_model']}")
    print(metrics["models"][metrics["selected_model"]])
