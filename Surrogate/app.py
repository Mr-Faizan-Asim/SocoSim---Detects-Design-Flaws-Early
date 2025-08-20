import streamlit as st
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ----------------- Define Model -----------------
class GCNModel(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=3, num_layers=3):
        super(GCNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x

# ----------------- Load Model -----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNModel().to(device)
model.load_state_dict(torch.load("surrogate_model.pth", map_location=device))
model.eval()

# ----------------- Streamlit UI -----------------
st.title("Surrogate Model: Static Structural Prediction with Mesh")

uploaded_file = st.file_uploader("Upload mesh JSON file", type="json")

if uploaded_file:
    mesh = json.load(uploaded_file)
    nodes = mesh["nodes"]
    elements = mesh["elements"]

    node_ids = sorted(nodes.keys())
    node_features = torch.tensor([nodes[n] for n in node_ids], dtype=torch.float32)

    # Build edges for model
    edge_index = []
    for elem_id, elem_nodes in elements:
        elem_nodes = [node_ids.index(str(n)) for n in elem_nodes]  # map to 0-based index
        for i in range(len(elem_nodes)):
            for j in range(len(elem_nodes)):
                if i != j:
                    edge_index.append([elem_nodes[i], elem_nodes[j]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    graph = Data(x=node_features, edge_index=edge_index)

    # Predict displacements
    with torch.no_grad():
        pred = model(graph.x.to(device), graph.edge_index.to(device)).cpu().numpy()

    orig_coords = node_features.numpy()
    displaced_coords = orig_coords + pred

    # ----------------- 3D Visualization with Edges -----------------
    fig = go.Figure()

    # Draw edges (lines) for original mesh
    for elem_id, elem_nodes in elements:
        elem_idx = [node_ids.index(str(n)) for n in elem_nodes]
        for i in range(len(elem_idx)):
            for j in range(i + 1, len(elem_idx)):
                fig.add_trace(go.Scatter3d(
                    x=[orig_coords[elem_idx[i], 0], orig_coords[elem_idx[j], 0]],
                    y=[orig_coords[elem_idx[i], 1], orig_coords[elem_idx[j], 1]],
                    z=[orig_coords[elem_idx[i], 2], orig_coords[elem_idx[j], 2]],
                    mode='lines',
                    line=dict(color='blue', width=3),
                    showlegend=False
                ))

    # Draw edges (lines) for displaced mesh
    for elem_id, elem_nodes in elements:
        elem_idx = [node_ids.index(str(n)) for n in elem_nodes]
        for i in range(len(elem_idx)):
            for j in range(i + 1, len(elem_idx)):
                fig.add_trace(go.Scatter3d(
                    x=[displaced_coords[elem_idx[i], 0], displaced_coords[elem_idx[j], 0]],
                    y=[displaced_coords[elem_idx[i], 1], displaced_coords[elem_idx[j], 1]],
                    z=[displaced_coords[elem_idx[i], 2], displaced_coords[elem_idx[j], 2]],
                    mode='lines',
                    line=dict(color='red', width=3),
                    showlegend=False
                ))

    # Draw nodes
    fig.add_trace(go.Scatter3d(
        x=orig_coords[:, 0], y=orig_coords[:, 1], z=orig_coords[:, 2],
        mode='markers', marker=dict(size=5, color='blue'),
        name='Original Nodes'
    ))
    fig.add_trace(go.Scatter3d(
        x=displaced_coords[:, 0], y=displaced_coords[:, 1], z=displaced_coords[:, 2],
        mode='markers', marker=dict(size=5, color='red'),
        name='Displaced Nodes'
    ))

    fig.update_layout(scene=dict(aspectmode='data'))
    st.plotly_chart(fig)
    st.write("Blue: Original mesh, Red: Predicted displaced mesh")

    # ----------------- Display Displacements as Table -----------------
    displacement_table = pd.DataFrame({
        "Node": node_ids,
        "u_x": pred[:, 0],
        "u_y": pred[:, 1],
        "u_z": pred[:, 2],
        "Displacement Magnitude": np.linalg.norm(pred, axis=1)
    })
    st.subheader("Predicted Displacements")
    st.dataframe(displacement_table)
