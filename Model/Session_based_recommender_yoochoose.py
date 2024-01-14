import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, TopKPooling
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score

# Function to calculate Mean Reciprocal Rank (MRR)
def calculate_mrr(probabilities, y_true):
    mrr = 0.0
    for prob, true_index in zip(probabilities, y_true):
        rank = np.argsort(-prob)[true_index]  # Rank of the true item
        mrr += 1.0 / (rank + 1)  # Reciprocal rank
    mrr /= len(y_true)  # Mean Reciprocal Rank
    return mrr

# Load data from CSV file
df = pd.read_csv('yoochoose.csv')  

# Create unique indices for users and items
user_indices = {user: index for index, user in enumerate(df['session_id'].unique())}
item_indices = {item: index for index, item in enumerate(df['item_id'].unique())}

# Map user and item indices to the DataFrame
df['user_index'] = df['session_id'].map(user_indices)
df['item_index'] = df['item_id'].map(item_indices)

# Create edge indices for the graph
edge_index = torch.tensor([df['user_index'].values, df['item_index'].values], dtype=torch.long)

# Create the session-based graph
data = Data(x=torch.zeros(len(user_indices) + len(item_indices), 1),
            edge_index=edge_index, y=None)

# Assuming you have sequences of item indices for each session
# Create sequences, for simplicity, we'll just consider the last item in each session
session_sequences = df.groupby('session_id')['item_index'].agg(list).reset_index()
session_sequences['target_item'] = session_sequences['item_index'].apply(lambda x: x[-1] if len(x) > 0 else None)

# Filter out sessions with only one item, as there's no "next item" to predict
session_sequences = session_sequences[session_sequences['item_index'].apply(len) > 1]

# Determine the maximum sequence length
max_sequence_length = session_sequences['item_index'].apply(len).max()

# Pad sequences to the maximum length
padded_sequences = torch.nn.utils.rnn.pad_sequence(
    [torch.tensor(seq, dtype=torch.long) for seq in session_sequences['item_index']],
    batch_first=True,
    padding_value=0  # Assuming 0 is not a valid item index
)

# Convert to PyTorch tensors
x_data = padded_sequences[:, :-1]  # Input sequences (exclude the last item)
y_data = padded_sequences[:, 1:]   # Target sequences (exclude the first item)

# Define the GNN model
class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GraphConv(1, 64)
        self.pool1 = TopKPooling(64, ratio=0.5)
        self.conv2 = GraphConv(64, 128)
        self.pool2 = TopKPooling(128, ratio=0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        return x

# Define a recommendation head
class RecommendationHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(RecommendationHead, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)

# Extend the GNN model
class GNNWithRecommendation(nn.Module):
    def __init__(self, gnn_model, input_size, hidden_size, output_size):
        super(GNNWithRecommendation, self).__init__()
        self.gnn_model = gnn_model
        self.recommendation_head = RecommendationHead(input_size, output_size)

    def forward(self, data):
        x = self.gnn_model(data)
        return self.recommendation_head(x)

# Initialize the extended model
gnn_model = GNNModel()
recommendation_model = GNNWithRecommendation(gnn_model, 128, 128, len(item_indices))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(recommendation_model.parameters(), lr=0.01)  # Adjust the learning rate

# Train the model
num_epochs = 20  # Increase the number of epochs
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    output = recommendation_model(data)

    # Flatten the output and target tensors
    output_flat = output.reshape(-1, len(item_indices))
    y_data_flat = y_data.reshape(-1)

    # Trim the output and target tensors to have the same number of elements
    min_elements = min(output_flat.size(0), y_data_flat.size(0))
    output_flat = output_flat[:min_elements, :]
    y_data_flat = y_data_flat[:min_elements]

    # Ensure the dimensions are compatible
    assert output_flat.size(0) == y_data_flat.size(0), "Output and target dimensions do not match"

    try:
        loss = criterion(output_flat, y_data_flat)
        loss.backward()
        optimizer.step()

        # Evaluate Recall
        y_pred = torch.argmax(output_flat, dim=1).detach().numpy()
        y_true = y_data_flat.numpy()

        recall = recall_score(y_true, y_pred, average='micro')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Recall: {recall}')

        # Calculate MRR
        probabilities = F.softmax(output_flat, dim=1).detach().numpy()
        mrr = calculate_mrr(probabilities, y_true)
        print(f'MRR: {mrr}')

    except Exception as e:
        print(f"Error during training: {e}")

# Function to get the recommendation for a given session
def get_recommendation(session, gnn_model, recommendation_model, item_indices):
    with torch.no_grad():
        # Convert the session to PyTorch tensor
        session_indices = [item_indices[item] for item in session]
        x_session = torch.tensor(session_indices, dtype=torch.long).view(1, -1)

        # Get the session embeddings using the GNN model
        session_embedding = gnn_model(data).detach()

        # Get the recommendation logits using the recommendation model
        logits = recommendation_model.recommendation_head(session_embedding)

        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=1).detach().numpy()

        # Get the index of the item with the highest probability (top recommendation)
        top_recommendation_index = np.argmax(probabilities)

        # Map the index back to the item
        top_recommendation = next(item for item, index in item_indices.items() if index == top_recommendation_index)

        return top_recommendation

# Example usage
session_to_predict = [214820942, 214530776, 214718203]  # Example session
predicted_item = get_recommendation(session_to_predict, gnn_model, recommendation_model, item_indices)

print(f"Predicted next item for session {session_to_predict}: {predicted_item}")
