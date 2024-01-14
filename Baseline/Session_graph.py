import pandas as pd
import networkx as nx
from sklearn.metrics import precision_score, recall_score

# Load the data from the CSV file
data = pd.read_csv("Diginetica.csv")

# Data Preprocessing and Cleaning
data.dropna(subset=['item_id'], inplace=True)  # Remove rows with missing item_id
data['item_id'] = data['item_id'].astype(int)  # Convert item_id to integer
data.sort_values(by=['session_id', 'eventdate', 'timeframe'], inplace=True)  # Sort data by session and time

# Create an empty directed graph to represent the session data
G = nx.DiGraph()

# Define a function to build the session graph
def build_session_graph(session_data, graph):
    previous_item = None
    for item_id in session_data['item_id']:
        graph.add_node(item_id)
        if previous_item:
            graph.add_edge(previous_item, item_id)
        previous_item = item_id

# Build the session graph
build_session_graph(data, G)

# Define a function to make recommendations based on the session graph
def recommend_items(session_graph, current_item, num_recommendations=10):
    recommendations = []
    for neighbor in session_graph.neighbors(current_item):
        recommendations.append(neighbor)
        if len(recommendations) >= num_recommendations:
            break
    return recommendations

# Define a function to calculate Precision, Recall, and MRR
def calculate_metrics(actual_items, recommended_items):
    actual_set = set(actual_items)
    recommended_set = set(recommended_items)

    # Precision
    precision = len(actual_set.intersection(recommended_set)) / len(recommended_set) if len(recommended_set) > 0 else 0.0

    # Recall
    recall = len(actual_set.intersection(recommended_set)) / len(actual_set) if len(actual_set) > 0 else 0.0

    # MRR
    mrr = 0.0
    for i, item in enumerate(recommended_items, 1):
        if item in actual_set:
            mrr = 1 / i
            break

    return precision, recall, mrr

# Evaluate the performance for each session
precision_list = []
recall_list = []
mrr_list = []

for session_id, session_data in data.groupby('session_id'):
    # Extract the last item in the session
    last_item_in_session = session_data['item_id'].iloc[-1]

    # Recommend items based on the session graph
    recommended_items = recommend_items(G, last_item_in_session)

    # Extract the actual items in the session
    actual_items = session_data['item_id'].tolist()

    # Calculate Precision, Recall, and MRR
    precision, recall, mrr = calculate_metrics(actual_items, recommended_items)

    precision_list.append(precision)
    recall_list.append(recall)
    mrr_list.append(mrr)

# Calculate the mean Precision, Recall, and MRR
mean_precision = sum(precision_list) / len(precision_list)
mean_recall = sum(recall_list) / len(recall_list)
mean_mrr = sum(mrr_list) / len(mrr_list)

print(f"Mean Precision: {mean_precision}")
print(f"Mean Recall: {mean_recall}")
print(f"Mean Reciprocal Rank (MRR): {mean_mrr}")
