# 📊 Customer Churn Prediction Web App (Flask + ML)

A complete **end-to-end Machine Learning web application** built with **Flask** that allows users to:

- Upload a CSV dataset
- Automatically clean and preprocess data
- Train a churn prediction model with **PCA**
- Save and load the trained model using **joblib**
- Make real-time churn predictions from a web interface

Designed to be **lightweight, Render-compatible, and beginner-friendly**.

---

## 🚀 Features

- 📁 CSV file upload
- 🧹 Automatic data cleaning
- 🔠 Automatic categorical encoding
- 📏 Feature scaling
- 📉 PCA for dimensionality reduction
- 🤖 ML model training (Logistic Regression)
- 💾 Model persistence with `joblib`
- 🔮 Real-time churn prediction
- 🌐 Bootstrap-based UI
- ☁️ Render deployment ready

---

## 🗂️ Project Structure

```text
ml_churn_project/
│
├── app.py                  # Main Flask application
├── model_trainer.py        # ML training logic
├── data_cleaner.py         # Automatic data cleaning
│
├── templates/
│   ├── upload.html         # File upload page
│   └── predict.html        # Prediction page
│
├── static/
│   └── style.css           # Optional styling
│
├── uploads/                # Uploaded CSV files
│
├── models/
│   └── churn_model.joblib  # Trained ML model
│
└── requirements.txt        # Python dependencies
````

---

## 🧠 Machine Learning Pipeline

1. **Data Cleaning**

   * Handles missing values
   * Removes duplicates

2. **Preprocessing**

   * Label encoding for categorical features
   * Standard scaling for numerical features

3. **Dimensionality Reduction**

   * PCA with 95% variance retention

4. **Model**

   * Logistic Regression (small, fast, Render-safe)

5. **Persistence**

   * Saved using `joblib` for compatibility and performance

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/ml_churn_project.git
cd ml_churn_project
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📤 How to Use

### 1️⃣ Upload Dataset

* Upload a CSV file containing customer data
* Dataset must include a target column named **`Churn`**

### 2️⃣ Train Model

* The model is trained automatically after upload
* PCA is applied to reduce dimensionality
* Model is saved to `models/churn_model.joblib`

### 3️⃣ Make Predictions

* Navigate to the prediction page
* Enter customer details
* Get churn prediction with probabilities

---

## 📄 Example Dataset Requirements

* CSV format
* Must contain a **target column** named `Churn`
* Categorical and numerical features supported
* Missing values handled automatically

---

## 🧪 API Endpoints

| Endpoint        | Method | Description              |
| --------------- | ------ | ------------------------ |
| `/`             | GET    | Upload page              |
| `/upload`       | POST   | Upload CSV & train model |
| `/predict_page` | GET    | Prediction form          |
| `/predict`      | POST   | Make churn prediction    |

---

## ☁️ Render Deployment Notes

* Uses `joblib` (fully supported on Render)
* Small model size (<5MB)
* No heavy dependencies
* Version-pinned libraries to avoid incompatibility

### Recommended start command:

```bash
python app.py
```

---

## 📌 requirements.txt

```txt
flask==3.0.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
joblib==1.3.2
```

---

## 🔐 Security Notes

* Uses `secure_filename` to prevent file upload attacks
* File size limited to 16MB
* Only `.csv` files accepted

---

## 🔮 Future Enhancements

* Dynamic form generation from model features
* Handling unseen categorical values
* Incremental training
* Model versioning
* Authentication & user management
* REST API documentation (Swagger)

---

## 👨‍💻 Author

Built with ❤️ using **Flask + scikit-learn**

---

## 📜 License

This project is open-source and free to use for learning and development purposes.

```


