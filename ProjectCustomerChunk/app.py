from flask import Flask, render_template, request
import pandas as pd
import os
from werkzeug.utils import secure_filename

from data_cleaner import clean_data, encode_and_scale
from model_trainer import apply_pca, train_model, save_model, load_model

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)

TARGET_COLUMN = "Churn"


@app.route("/")
def index():
    return render_template("upload.html")


# -------------------------
# UPLOAD & TRAIN (NO JS)
# -------------------------
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return render_template("upload.html", error="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("upload.html", error="No file selected")

        if not file.filename.endswith(".csv"):
            return render_template("upload.html", error="Only CSV files are allowed")

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            secure_filename(file.filename)
        )
        file.save(filepath)

        df = pd.read_csv(filepath)
        df = clean_data(df)

        if TARGET_COLUMN not in df.columns:
            return render_template(
                "upload.html",
                error=f"Target column '{TARGET_COLUMN}' not found in CSV"
            )

        X, y, scaler, encoders = encode_and_scale(df, TARGET_COLUMN)
        feature_names = X.columns.tolist()

        X_pca, pca = apply_pca(X)
        model, accuracy = train_model(X_pca, y)

        save_model({
            "model": model,
            "pca": pca,
            "scaler": scaler,
            "label_encoders": encoders,
            "feature_names": feature_names
        })

        return render_template(
            "upload.html",
            success=True,
            accuracy=f"{accuracy:.2%}",
            features_original=len(feature_names),
            features_pca=X_pca.shape[1],
            samples=len(df)
        )

    except Exception as e:
        return render_template("upload.html", error=str(e))


# -------------------------
# SHOW PREDICT PAGE
# -------------------------
@app.route("/predict_page")
def predict_page():
    if not load_model():
        return "Train model first", 400
    return render_template("predict.html")


# -------------------------
# PREDICT (NO JS, FORM)
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    package = load_model()
    if not package:
        return render_template("predict.html", error="Model not found")

    # Get form input
    input_data = dict(request.form)
    df = pd.DataFrame([input_data])

    # Ensure all features exist
    for col in package["feature_names"]:
        if col not in df.columns:
            df[col] = 0

    df = df[package["feature_names"]]

    # Encode categorical features
    for col, enc in package["label_encoders"].items():
        if col != "target" and col in df.columns:
            try:
                df[col] = enc.transform(df[col].astype(str))
            except:
                df[col] = 0

    # Scale + PCA
    scaled = package["scaler"].transform(df)
    pca_data = package["pca"].transform(scaled)

    pred = package["model"].predict(pca_data)[0]
    prob = package["model"].predict_proba(pca_data)[0]

    if "target" in package["label_encoders"]:
        pred = package["label_encoders"]["target"].inverse_transform([pred])[0]

    return render_template(
        "predict.html",
        prediction=str(pred),
        prob0=round(prob[0] * 100, 1),
        prob1=round(prob[1] * 100, 1)
    )


if __name__ == "__main__":
    app.run(debug=True)
