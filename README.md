# 🕵️ Fraud Detection System

A Streamlit-powered web application for detecting potentially fraudulent transactions in real time and in batch mode using a trained machine learning pipeline.

This project was developed to demonstrate practical fraud detection using machine learning models wrapped in an interactive web app.

---

## 🚀 Features

* **Real-Time Prediction**

  * Analyze a single transaction by inputting customer, merchant, and transaction details.
  * Get fraud probability, risk level, and recommended actions.

* **Batch Processing**

  * Upload a CSV file of transactions.
  * Get fraud probability for each transaction, with downloadable results.
  * Summary statistics: total transactions, high-risk counts, and average fraud probability.

* **Interactive Visuals**

  * Fraud probability displayed with metrics and a dynamic gauge chart.
  * Tabular results with sorting and filtering in Streamlit.

---

## 🛠️ Tech Stack

* **Frontend / Dashboard**: [Streamlit](https://streamlit.io/)
* **Backend**: Python
* **Machine Learning Pipeline**: Scikit-learn / Joblib
* **Data Handling**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn, PyDeck

---

## 📂 Project Structure

```
fraud-detection-system/
│── fraud_detection_pipeline.pkl   # Trained ML pipeline (must be present)
│── fraud.py                       # Main Streamlit application
│── requirements.txt               # Python dependencies
│── README.md                      # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/fraud-detection-system.git
   cd fraud-detection-system
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure you have the trained pipeline**

   * Place `fraud_detection_pipeline.pkl` in the project directory.
   * If missing, retrain or request the model file.

---

## ▶️ Running the App

The app is be available at:
👉 `frauddetectapp.streamlit.app`

---

## 📊 Input Data Format

For **Batch Processing**, your CSV file must include these columns:

```
merchant, category, amt, last, gender, 
lat, long, city_pop, job, merch_lat, 
merch_long, hour, month
```

✔ Example row:

```
M12345, grocery, 150.0, 7, 1, 40.71, -74.01, 1000000, Engineer, 40.72, -74.02, 14, 8
```

---

## 🧠 Fraud Risk Levels

| Probability Range | Risk Level | Recommended Action                                                 |
| ----------------- | ---------- | ------------------------------------------------------------------ |
| **> 70%**         | High       | Verify identity, request additional authentication, review history |
| **30–70%**        | Medium     | Send verification code, monitor transactions                       |
| **< 30%**         | Low        | No action needed                                                   |

---

## 📝 Future Improvements

* Add authentication & role-based access.
* Expand feature engineering for better predictions.
* Integrate live transaction APIs.
* Deploy app on **Streamlit Cloud** or **Heroku**.

---

## 👨‍💻 Author

* **Kolaru Gideon Mosimiloluwa**

---
