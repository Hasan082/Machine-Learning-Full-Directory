from flask import Flask, render_template, request
import pandas as pd
import os
from werkzeug.utils import secure_filename

from data_cleaning import clean_data, encode_data
from models_trainer import apply_pca, train_models, save_models, load_models

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)

TARGET_COLUMN = "Churn"


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template('upload.html')
    
    try:
        if 'file' not in request.files:
            return render_template('upload.html', error="No file uploaded")

        file = request.files['file']

        if not file.filename.endswith(".csv"):
            return render_template('upload.html', error="Only CSV files are allowed")

        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            secure_filename(file.filename)
        )

        file.save(file_path)

        df = pd.read_csv(file_path)
        df = clean_data(df)

        if TARGET_COLUMN not in df.columns:
            return render_template(
                'upload.html',
                error=f"Target column '{TARGET_COLUMN}' not found."
            )

        # Encode + scale
        X, y, scaler, encoder = encode_data(df, TARGET_COLUMN)

        # ✅ FIXED: Track the feature names AFTER encoding (what the scaler was fitted on)
        feature_names = X.columns.to_list()

        # PCA + Model
        X_pca, pca = apply_pca(X)
        model, accuracy = train_models(X_pca, y)

        save_models({
            "model": model,
            "pca": pca,
            "scaler": scaler,
            "label_encoder": encoder,
            "feature_names": feature_names,  # Features after encoding
        })

        return render_template(
            "upload.html",
            success=True,
            accuracy=f"{accuracy:.2f}",
            feature_original=len(feature_names),
            feature_pca=X_pca.shape[1],
            samples=len(df)
        )

    except Exception as err:
        return render_template("upload.html", error=str(err))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template('predict.html')
    
    package = load_models()
    if not package:
        return render_template('predict.html', error="Please train the model first by uploading data")

    try:
        # Get form input
        input_data = dict(request.form)
        df = pd.DataFrame([input_data])

        # 1️⃣ Align columns with training data (before encoding)
        df = df.reindex(columns=package["feature_names"], fill_value=0)

        # 2️⃣ Encode categorical features
        for col, enc in package["label_encoder"].items():
            if col in df.columns:
                try:
                    df[col] = enc.transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    df[col] = 0

        # 3️⃣ Convert all columns to numeric (safety check)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 4️⃣ ✅ FIXED: Scale ALL features (same as during training)
        # The scaler was fitted on all encoded features, so we scale all of them
        df_scaled = pd.DataFrame(
            package["scaler"].transform(df),
            columns=df.columns
        )

        # 5️⃣ PCA transform
        X_pca = package["pca"].transform(df_scaled)

        # 6️⃣ Predict
        pred = package["model"].predict(X_pca)[0]
        prob = package["model"].predict_proba(X_pca)[0]
        
        if prob[1] < 0.30:
            risk = "Low"
            comment = "Customer is likely to stay. No immediate action required."
        elif prob[1] < 0.60:
            risk = "Medium"
            comment = "Customer shows signs of churn. Consider proactive engagement."
        else:
            risk = "High"
            comment = "Customer is likely to churn. Immediate retention action recommended."

        

        return render_template(
            "predict.html",
            prediction="Churn" if pred == 1 else "Not Churn",
            prob0=round(prob[0] * 100, 1),
            prob1=round(prob[1] * 100, 1),
            risk=risk,
            comment=comment
        )

    except Exception as err:
        return render_template("predict.html", error=f"Prediction error: {str(err)}")


if __name__ == "__main__":
    app.run(debug=True)