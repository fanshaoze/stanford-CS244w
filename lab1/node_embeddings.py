import networkx as nx
import torch
import random
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub

from sklearn.decomposition import PCA
import torch.nn as nn
from torch.optim import SGD
from torch.nn import Linear
from torch_geometric.nn import GCNConv


def average_degree(num_edges, num_nodes):
    # and number of nodes, and returns the average node degree of
    # the graph. Round the result to nearest integer (for example
    # 3.3 will be rounded to 3 and 3.7 will be rounded to 4)

    avg_degree = 0

    ############# Your code here ############
    avg_degree = num_edges * 2 / num_nodes
    #########################################

    return round(avg_degree)


def average_clustering_coefficient(G):
    # and returns the average clustering coefficient. Round
    # the result to 2 decimal places (for example 3.333 will
    # be rounded to 3.33 and 3.7571 will be rounded to 3.76)

    avg_cluster_coef = 0

    ############# Your code here ############
    ## Note:
    ## 1: Please use the appropriate NetworkX clustering function
    coefficients = nx.clustering(G)
    avg_cluster_coef = sum(list(coefficients.values())) / G.number_of_nodes()
    avg_cluster_coef = round(avg_cluster_coef, 2)
    #########################################

    return avg_cluster_coef


def one_iter_pagerank(G, beta, r0, node_id):
    # Implement this function that takes a nx.Graph, beta, r0 and node id.
    # The return value r1 is one interation PageRank value for the input node.
    # Please round r1 to 2 decimal places.

    r1 = 0

    ############# Your code here ############
    ## Note:
    ## 1: You should not use nx.pagerank
    r1 = round(sum([beta * r0 / G.degree[nei] for nei in G.neighbors(node_id)]) +
               (1 - beta) * 1 / G.number_of_edges(), 2)  # add 1/N for each node, not each neighbor
    #########################################

    return r1


def closeness_centrality(G, node=5):
    # Implement the function that calculates closeness centrality
    # for a node in karate club network. G is the input karate club
    # network and node is the node id in the graph. Please round the
    # closeness centrality result to 2 decimal places.

    closeness = 0

    ## Note:
    ## 1: You can use networkx closeness centrality function.
    ## 2: Notice that networkx closeness centrality returns the normalized
    ## closeness directly, which is different from the raw (unnormalized)
    ## one that we learned in the lecture.
    # closeness = nx.closeness_centrality(G, node)
    print(closeness)
    closeness = 1 / sum(list(nx.single_source_shortest_path_length(G=G, source=node).values()))
    # print(sum([1 for i in list(nx.single_source_shortest_path_length(G=G, source=node).values()) if i != 0])
    #       / sum(list(nx.single_source_shortest_path_length(G=G, source=node).values())))

    #########################################

    return round(closeness, 2)


def torch_test():
    # Generate 3 x 4 tensor with all ones
    ones = torch.ones(3, 4)
    print(ones)

    # Generate 3 x 4 tensor with all zeros
    zeros = torch.zeros(3, 4)
    print(zeros)

    # Generate 3 x 4 tensor with random values on the interval [0, 1)
    random_tensor = torch.rand(3, 4)
    print(random_tensor)

    # Get the shape of the tensor
    print(ones.shape)

    # Create a 3 x 4 tensor with all 32-bit floating point zeros
    zeros = torch.zeros(3, 4, dtype=torch.float32)
    print(zeros.dtype)

    # Change the tensor dtype to 64-bit integer
    zeros = zeros.type(torch.long)
    print(zeros.dtype)


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
def embedding_test():
    emb_sample = nn.Embedding(num_embeddings=4, embedding_dim=8)
    print('Sample embedding layer: {}'.format(emb_sample))

    # Select an embedding in emb_sample
    id = torch.LongTensor([1])
    print(emb_sample(id))

    # Select multiple embeddings
    ids = torch.LongTensor([1, 3])
    print(emb_sample(ids))

    # Get the shape of the embedding weight matrix
    shape = emb_sample.weight.data.shape
    print(shape)

    # Overwrite the weight to tensor with all ones
    emb_sample.weight.data = torch.ones(shape)

    # Let's check if the emb is indeed initilized
    ids = torch.LongTensor([0, 3])
    print(emb_sample(ids))


torch.manual_seed(1)


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


def visualize_emb(emb):
    X = emb.weight.data.numpy()
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure(figsize=(6, 6))
    club1_x = []
    club1_y = []
    club2_x = []
    club2_y = []
    for node in G.nodes(data=True):
        if node[1]['club'] == 'Mr. Hi':
            club1_x.append(components[node[0]][0])
            club1_y.append(components[node[0]][1])
        else:
            club2_x.append(components[node[0]][0])
            club2_y.append(components[node[0]][1])
    plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
    plt.scatter(club2_x, club2_y, color="blue", label="Officer")
    plt.legend()
    # plt.show()


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


def train(emb, loss_fn, sigmoid, train_label, train_edge):
    # TODO: Train the embedding layer here. You can also change epochs and
    # learning rate. In general, you need to implement:
    # (1) Get the embeddings of the nodes in train_edge
    # (2) Dot product the embeddings between each node pair
    # (3) Feed the dot product result into sigmoid
    # (4) Feed the sigmoid output into the loss_fn
    # (5) Print both loss and accuracy of each epoch
    # (6) Update the embeddings using the loss and optimizer
    # (as a sanity check, the loss should decrease during training)

    epochs = 500
    learning_rate = 0.1

    optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)

    for i in range(epochs):
        optimizer.zero_grad()
        ############# Your code here ############
        emb_start = emb(train_edge[0])
        emb_end = emb(train_edge[1])

        # print(emb_start, emb_end)
        # pred = torch.dot(emb_start, emb_end)

        pred = torch.sum(emb_start * emb_end, dim=-1) #TODO: Why
        pred = sigmoid(pred)
        loss = loss_fn(pred, train_label)
        loss.backward()
        optimizer.step()

        print(f"{i} epoch Loss: {loss}, accuracy: {accuracy(pred=pred, label=train_label)}")

        #########################################


G = nx.karate_club_graph()

# G is an undirected graph
print(type(G))
# Visualize the graph
nx.draw(G, with_labels=True)
# plt.show()

num_edges = G.number_of_edges()
num_nodes = G.number_of_nodes()
avg_degree = average_degree(num_edges, num_nodes)
print(f"Average degree of karate club network is {avg_degree}")

avg_cluster_coef = average_clustering_coefficient(G)
print("Average clustering coefficient of karate club network is {}".format(avg_cluster_coef))

beta = 0.8
r0 = 1 / G.number_of_nodes()
node = 0
r1 = one_iter_pagerank(G, beta, r0, node)
print("The PageRank value for node 0 after one iteration is {}".format(r1))

closeness = closeness_centrality(G, node=5)
print("The karate club network has closeness centrality {}".format(closeness))

torch_test()

pos_edge_list = graph_to_edge_list(G)
pos_edge_index = edge_list_to_tensor(pos_edge_list)
print("The pos_edge_index tensor has shape {}".format(pos_edge_index.shape))
print("The pos_edge_index tensor has sum value {}".format(torch.sum(pos_edge_index)))

# Sample 78 negative edges
neg_edge_list = sample_negative_edges(G, len(pos_edge_list))

# Transform the negative edge list to tensor
neg_edge_index = edge_list_to_tensor(neg_edge_list)
print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))

embedding_test()

emb = create_node_emb()
ids = torch.LongTensor([0, 3])

# Print the embedding layer
print("Embedding: {}".format(emb))

# An example that gets the embeddings for node 0 and 3
print(emb(ids))

visualize_emb(emb)

loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()

print(pos_edge_index.shape)

# Generate the positive and negative labels
pos_label = torch.ones(pos_edge_index.shape[1], )
neg_label = torch.zeros(neg_edge_index.shape[1], )

# Concat positive and negative labels into one tensor
train_label = torch.cat([pos_label, neg_label], dim=0)

# Concat positive and negative edges into one tensor
# Since the network is very small, we do not split the edges into val/test sets
train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
print(train_edge.shape)

train(emb, loss_fn, sigmoid, train_label, train_edge)
