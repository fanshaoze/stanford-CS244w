import csv
import json

import networkx as nx
import torch
import random

from torch_geometric.datasets import KarateClub

from sklearn.decomposition import PCA
import torch.nn as nn
from torch.optim import SGD
from torch.nn import Linear
from torch_geometric.nn import GCNConv


def single_node_label(G, node, pred_label='clossness_centrality'):
    # Implement the function that calculates closeness centrality
    # for a node in karate club network. G is the input karate club
    # network and node is the node id in the graph. Please round the
    # closeness centrality result to 2 decimal places.

    y = 0

    # closeness = nx.closeness_centrality(G, node)
    if pred_label == 'clossness_centrality':
        y = 1 / sum(list(nx.single_source_shortest_path_length(G=G, source=node).values()))
        # print(f'closeness of node {node} is {y}')
    return y


def global_nodes_label(G, pred_feature):
    return [single_node_label(G, node, pred_feature) for node in G.nodes()]


def global_node_feature(G):
    return [sum(list(nx.single_source_shortest_path_length(G=G, source=_node).values())) for _node in G.nodes()]


def graph_to_edge_list(G):
    # an nx.Graph. The returned edge_list should be a list of tuples
    # where each tuple is a tuple representing an edge connected
    # by two nodes.

    edge_list = []

    ############# Your code here ############
    edge_list = list(G.edges())
    #########################################

    return edge_list


def edge_list_to_tensor(edge_list):
    # TODO: Implement the function that transforms the edge_list to
    # tensor. The input edge_list is a list of tuples and the resulting
    # tensor should have the shape [2 x len(edge_list)].

    edge_index = torch.tensor([])

    ############# Your code here ############
    edge_index = torch.tensor(data=edge_list, dtype=torch.long)
    #########################################

    return edge_index.T


def sample_negative_edges(G, num_neg_samples):
    # TODO: Implement the function that returns a list of negative edges.
    # The number of sampled negative edges is num_neg_samples. You do not
    # need to consider the corner case when the number of possible negative edges
    # is less than num_neg_samples. It should be ok as long as your implementation
    # works on the karate club network. In this implementation, self loops should
    # not be considered as either a positive or negative edge. Also, notice that
    # the karate club network is an undirected graph, if (0, 1) is a positive
    # edge, do you think (1, 0) can be a negative one?

    neg_edge_list = []

    ############# Your code here ############
    edges = G.edges()
    rand_nodes = list(G.nodes())
    random.shuffle(rand_nodes)
    for start in rand_nodes:
        for end in rand_nodes:
            if (start < end) and ((start, end) not in edges):
                neg_edge_list.append((start, end))
                num_neg_samples -= 1
                if not num_neg_samples:
                    return neg_edge_list
    #########################################


# Initialize an embedding layer
# Suppose we want to have embedding for 4 items (e.g., nodes)
# Each item is represented with 8 dimensional vector
_seed = 5
torch.manual_seed(_seed)

# task, edge prediction: coefficient, random

def create_node_emb(num_node=34, _embedding_dim=16):
    # TODO: Implement this function that will create the node embedding matrix.
    # A torch.nn.Embedding layer will be returned. You do not need to change
    # the values of num_node and embedding_dim. The weight matrix of returned
    # layer should be initialized under uniform distribution.

    emb = None

    ############# Your code here ############
    emb = nn.Embedding(num_embeddings=num_node, embedding_dim=_embedding_dim)
    emb.weight.data = torch.rand(emb.weight.data.shape)
    #########################################

    return emb


def accuracy(pred, label):
    # finished: Implement the accuracy function. This function takes the
    # pred tensor (the resulting tensor after sigmoid) and the label
    # tensor (torch.LongTensor). Predicted value greater than 0.5 will
    # be classified as label 1. Else it will be classified as label 0.
    # The returned accuracy should be rounded to 4 decimal places.
    # For example, accuracy 0.82956 will be rounded to 0.8296.

    accu = 0.0

    ############# Your code here ############
    preds = [1 if p > 0.5 else 0 for p in pred]
    assert len(preds) == len(label)
    accu = sum([1 if preds[i] == label[i] else 0 for i in range(len(preds))]) / len(preds)

    #########################################

    return round(accu, 4)


def train(emb, loss_fn, active, train_label, train_nodes):
    # TODO: Train the embedding layer here. You can also change epochs and
    # learning rate. In general, you need to implement:
    # (1) Get the embeddings of the nodes in train_edge
    # (2) Dot product the embeddings between each node pair
    # (3) Feed the dot product result into sigmoid
    # (4) Feed the sigmoid output into the loss_fn
    # (5) Print both loss and accuracy of each epoch
    # (6) Update the embeddings using the loss and optimizer
    # (as a sanity check, the loss should decrease during training)
    plot_x = []
    plot_y = []

    epochs = 10000
    learning_rate = 0.1

    optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)

    for i in range(epochs):
        optimizer.zero_grad()
        ############# Your code here ############
        # emb_start = emb(train_edge[0])
        # emb_end = emb(train_edge[1])
        node_emb = emb(train_nodes)


        # print(emb_start, emb_end)
        # pred = torch.dot(emb_start, emb_end)
        pred = torch.squeeze(node_emb)
        if active:
            pred = active(pred)
        loss = loss_fn(pred, train_label)
        loss.backward()
        optimizer.step()

        print(f"{i} epoch Loss: {loss}, accuracy: {accuracy(pred=pred, label=train_label)}")
        plot_x.append(i)
        plot_y.append(float(loss.detach().numpy()))

    # plt.plot(plot_x, plot_y)

    # plt.savefig("practice_Sig.png", format="png")
    # plt.close()
    return plot_x, plot_y


G = nx.karate_club_graph()

# G is an undirected graph
print(type(G))
# Visualize the graph
nx.draw(G, with_labels=True)
# plt.show()

num_edges = G.number_of_edges()
num_nodes = G.number_of_nodes()
nodes = G.nodes()

beta = 0.8
r0 = 1 / G.number_of_nodes()
node = 0

pos_edge_list = graph_to_edge_list(G)
pos_edge_index = edge_list_to_tensor(pos_edge_list)
print("The pos_edge_index tensor has shape {}".format(pos_edge_index.shape))
print("The pos_edge_index tensor has sum value {}".format(torch.sum(pos_edge_index)))

# Sample 78 negative edges

emb = create_node_emb(num_node=120, _embedding_dim=1)
ids = torch.LongTensor([0, 3])

# Print the embedding layer
print("Embedding: {}".format(emb))

# An example that gets the embeddings for node 0 and 3
print(emb(ids))

# visualize_emb(emb)

loss_fn = nn.BCELoss()
active = nn.Sigmoid()
# active = nn.Tanh()

print(pos_edge_index.shape)

# Generate the positive and negative labels
# pos_label = torch.ones(pos_edge_index.shape[1], )
# neg_label = torch.zeros(neg_edge_index.shape[1], )


# Concat positive and negative labels into one tensor
# train_label = torch.cat([pos_label, neg_label], dim=0)
train_label = torch.FloatTensor(global_nodes_label(G, 'clossness_centrality'))
train_nodes = torch.IntTensor(G.nodes)
# train_nodes = torch.IntTensor(global_node_feature(G))
# Concat positive and negative edges into one tensor
# Since the network is very small, we do not split the edges into val/test sets
# train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
# print(train_edge.shape)

steps, loss = train(emb, loss_fn, active, train_label, train_nodes=train_nodes)
json.dump([steps, loss], open(f'practice_feature_{_seed}.json', 'w'))
