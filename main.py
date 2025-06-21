from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "GraphSAGE Fraud Detection API is live!"}


# Define the model architecture
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model
in_channels = 30         # Replace with your actual number of node features
hidden_channels = 32
out_channels = 2         # Binary classification

model = GraphSAGEModel(in_channels, hidden_channels, out_channels)
model.load_state_dict(torch.load("graphsage_fraud_model.pt", map_location=torch.device("cpu")))
model.eval()

# Define request body
class TransactionData(BaseModel):
    node_features: list[list[float]]  # 2D list of node features
    edge_index: list[list[int]]       # 2D list of edges

@app.post("/predict")
def predict(data: TransactionData):
    try:
        # Convert inputs to tensors
        x = torch.tensor(data.node_features, dtype=torch.float)
        edge_index = torch.tensor(data.edge_index, dtype=torch.long)

        # Run inference
        with torch.no_grad():
            pred = model(x, edge_index)
            prediction = torch.argmax(pred, dim=1).tolist()

        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}
