import os
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def apply_pca(X, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    X_pca = pca.fit_transform(X)
    print(f"📉 PCA: {X.shape[1]} → {X_pca.shape[1]}")
    return X_pca, pca


def train_model(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"🎯 Accuracy: {acc:.2%}")

    return model, acc


def save_model(package, filename="churn_model.joblib"):
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(package, path, compress=3)
    print(f"💾 Saved model → {path}")
    return path


def load_model(filename="churn_model.joblib"):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None
