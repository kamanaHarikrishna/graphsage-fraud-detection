# 💳 GraphSAGE-based Credit Card Fraud Detection

This project implements an **AI-powered credit card fraud detection system** that combines traditional machine learning (XGBoost) with a powerful **Graph Neural Network (GraphSAGE)** model to identify fraudulent transactions based on graph-structured data.

## 🎯 Objective

The goal is to improve fraud detection by leveraging **graph relationships** between entities (customers, merchants, cards, devices, etc.) using **Graph Neural Networks**, providing better accuracy than conventional models.

---

## ⚙️ Technologies & Frameworks

- Python
- PyTorch & PyTorch Geometric
- XGBoost
- Scikit-learn
- NetworkX
- Pandas, NumPy
- FastAPI (for model deployment)
- Streamlit (for visualization)

---

## 🧠 Models Implemented

- ✅ **XGBoost**: Baseline model using tabular data
- ✅ **GraphSAGE**: Semi-supervised GNN model leveraging structural transaction patterns

---

## 🧪 Workflow

1. **Data Preprocessing**:
   - Null value handling
   - Normalization
   - Feature engineering

2. **Graph Construction**:
   - Nodes: Transactions, Cards, Merchants
   - Edges: Represent relationships (e.g., shared cards, merchant-device pairs)

3. **Model Training**:
   - Train XGBoost on preprocessed features
   - Train GraphSAGE on graph data

4. **Evaluation**:
   - Accuracy, Precision, Recall, F1 Score, ROC-AUC

5. **Deployment**:
   - FastAPI endpoint for fraud prediction
   - Optional Streamlit dashboard for visualizing patterns

---

## 📁 Project Structure

graphsage-fraud-detection/
│
├── data/
│ └── creditcard.csv # Transaction dataset
│
├── models/
│ ├── xgboost_model.pkl # Trained XGBoost model
│ └── graphsage_fraud_model.pt # Trained GraphSAGE model
│
├── app.py # FastAPI app for model inference
├── visualize.py # Streamlit app for dashboard (optional)
├── preprocess.py # Data preprocessing pipeline
├── graph_constructor.py # Graph generation script
├── train_xgboost.py # Train and evaluate XGBoost
├── train_graphsage.py # Train GraphSAGE model
├── requirements.txt # All dependencies
└── README.md # Project documentation


---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kamanaHarikrishna/graphsage-fraud-detection.git
   cd graphsage-fraud-detection
🧑‍💻 Author
Hari Krishna Kamana
🔗 LinkedIn

