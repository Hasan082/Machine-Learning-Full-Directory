import os
import joblib 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


def apply_pca(X, n_components=0.95):
    """
    Apply PCA for dimensionality reduction.
    
    Args:
        X: Feature matrix (already encoded and scaled)
        n_components: Number of components or variance to retain (default: 0.95)
    
    Returns:
        X_pca: Transformed feature matrix
        pca: Fitted PCA object
    """
    print(f"Applying PCA with n_components={n_components}")
    pca = PCA(n_components=n_components, svd_solver='full')
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced features from {X.shape[1]} to {X_pca.shape[1]}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return X_pca, pca


def train_models(X, y, test_size=0.2):
    """
    Train a Logistic Regression model.
    
    Args:
        X: Feature matrix (PCA-transformed)
        y: Target variable
        test_size: Proportion of data for testing (default: 0.2)
    
    Returns:
        model: Trained model
        acc: Accuracy score on test set
    """
    print("Training Logistic Regression model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Model trained! Accuracy: {acc:.4f}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return model, acc
    
    
def save_models(package, file_name="models.joblib"):
    """
    Save the trained model and preprocessing objects.
    
    Args:
        package: Dictionary containing model, scaler, encoders, etc.
        file_name: Name of the file to save (default: "models.joblib")
    
    Returns:
        path: Path where the model was saved
    """
    path = os.path.join(MODEL_DIR, file_name)
    joblib.dump(package, path)
    print(f"✅ Model package saved at {path}")
    print(f"Package contents: {list(package.keys())}")
    
    return path


def load_models(file_name="models.joblib"):
    """
    Load the trained model and preprocessing objects.
    
    Args:
        file_name: Name of the file to load (default: "models.joblib")
    
    Returns:
        package: Dictionary containing model, scaler, encoders, etc., or None if not found
    """
    path = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(path):
        print(f"Loading model from {path}")
        package = joblib.load(path)
        print(f"Package loaded with keys: {list(package.keys())}")
        return package
    else:
        print(f"⚠️ Model file not found at {path}")
        return None