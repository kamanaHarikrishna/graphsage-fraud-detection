# 💳 GraphSAGE-based Credit Card Fraud Detection

This project implements an **AI-powered credit card fraud detection system** that combines traditional machine learning (XGBoost) with a powerful **Graph Neural Network (GraphSAGE)** model to identify fraudulent transactions using both tabular and graph-structured data.

---

## 🎯 Objective

To enhance fraud detection by exploiting hidden relationships in transaction networks using **Graph Neural Networks (GNNs)**, particularly **GraphSAGE**, alongside XGBoost for comparison.

---

## ⚙️ Technologies & Tools

- **Programming**: Python  
- **ML/DL**: XGBoost, Scikit-learn, PyTorch  
- **GNN**: PyTorch Geometric (GraphSAGE)  
- **Data Processing**: Pandas, NumPy, NetworkX  
- **Deployment**: FastAPI (backend API), Streamlit (frontend UI)

---

## 🧠 Models Implemented

- ✅ **XGBoost** — For baseline fraud detection on tabular data  
- ✅ **GraphSAGE** — For advanced fraud detection leveraging graph topology

---

## 🔄 Workflow

1. **Data Preprocessing**  
   - Clean and normalize transaction data  
   - Feature engineering

2. **Graph Construction**  
   - Build graph with nodes as users, cards, merchants, devices  
   - Create edges representing relationships (e.g., same IP or device used)

3. **Model Training**  
   - Train baseline XGBoost classifier  
   - Train GraphSAGE GNN model with edge embeddings

4. **Evaluation**  
   - Compare model performance using metrics

5. **Deployment**  
   - FastAPI endpoint for predictions  
   - Optional Streamlit dashboard to visualize insights

---

## 📁 Directory Structure

```
graphsage-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── xgboost_model.pkl
│   └── graphsage_fraud_model.pt
│
├── app.py                   # FastAPI app
├── visualize.py             # Streamlit dashboard
├── preprocess.py            # Cleaning & feature engineering
├── graph_constructor.py     # Create graph structure
├── train_xgboost.py         # Train XGBoost model
├── train_graphsage.py       # Train GNN model
├── requirements.txt         # Dependencies
└── README.md
```

---

## 🚀 Getting Started

```bash
# 1. Clone repository
git clone https://github.com/kamanaHarikrishna/graphsage-fraud-detection.git
cd graphsage-fraud-detection

# 2. (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate     # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models
python train_xgboost.py
python train_graphsage.py

# 5. Run FastAPI backend
uvicorn app:app --reload

# 6. (Optional) Launch Streamlit UI
streamlit run visualize.py
```

---

## 📊 Metrics Used

- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

---

## 📌 Use Case

This can be integrated into real-time fraud monitoring systems by processing live transaction data, constructing on-the-fly graphs, and applying the trained models to flag suspicious transactions.

---

## 🙋‍♂️ Author

**Hari Krishna Kamana**  
🔗 [LinkedIn](https://www.linkedin.com/in/kamanaharikrishna)

---



