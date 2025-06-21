import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import ast
from PIL import Image
import base64

# Set background image
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the background setup
set_background("background.png")  # Ensure this file exists in the same folder

# Define the GraphSAGE model class
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

# Load model
in_channels = 30
hidden_channels = 32
out_channels = 2
model = GraphSAGEModel(in_channels, hidden_channels, out_channels)
model.load_state_dict(torch.load("graphsage_fraud_model.pt", map_location=torch.device("cpu")))
model.eval()

# Page title
st.markdown("<h1 style='color:#FF5733;'>GraphSAGE Credit Card Fraud Detection</h1>", unsafe_allow_html=True)

# User input
st.markdown("<h3 style='color:#FF5733;'>Enter Node Features</h3>", unsafe_allow_html=True)
node_input = st.text_area("(comma-separated per row)", height=100)

st.markdown("<h3 style='color:#FF5733;'>Enter Edge Index</h3>", unsafe_allow_html=True)
edge_input = st.text_area("(comma-separated pairs per row)", height=100)

if st.button("Predict Fraud"):
    try:
        node_features = ast.literal_eval(node_input)
        edge_index = ast.literal_eval(edge_input)

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        with torch.no_grad():
            prediction = model(x, edge_index)
            predicted_labels = torch.argmax(prediction, dim=1).tolist()

        st.success("Prediction completed successfully!")
        st.markdown("### Predicted Fraud Labels:")
        st.code(predicted_labels)
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("<hr style='margin-top: 50px;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#CCCCCC;'>Developed by Hari</p>", unsafe_allow_html=True)
