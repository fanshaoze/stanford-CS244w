import json

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
from lab1.arguments import get_args
import torch.nn.functional as F


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


# def visualize_emb(emb):
#     X = emb.weight.data.numpy()
#     pca = PCA(n_components=2)
#     components = pca.fit_transform(X)
#     plt.figure(figsize=(6, 6))
#     club1_x = []
#     club1_y = []
#     club2_x = []
#     club2_y = []
#     for node in G.nodes(data=True):
#         if node[1]['club'] == 'Mr. Hi':
#             club1_x.append(components[node[0]][0])
#             club1_y.append(components[node[0]][1])
#         else:
#             club2_x.append(components[node[0]][0])
#             club2_y.append(components[node[0]][1])
#     plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
#     plt.scatter(club2_x, club2_y, color="blue", label="Officer")
#     plt.legend()
#     # plt.show()


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


def train(emb, hpara, loss_fn, sigmoid, train_label, train_edge, dev_label, dev_edge, test_label, test_edge):
    """

    :param emb:
    :param hpara:
    :param loss_fn:
    :param sigmoid:
    :param train_label:
    :param train_edge:
    :param validate_label:
    :param validate_edge:
    :param test_label:
    :param test_edge:
    :return:
    """

    epochs = hpara.epoch
    # TODO: early stopping, learning_rate,
    learning_rate = hpara.lr

    optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)
    best_dev_metric = 100
    patience_counter = 0
    for ep in range(epochs):
        optimizer.zero_grad()
        ############# Your code here ############
        # same env, same embeddings for same input
        emb_start = emb(train_edge[0])
        emb_end = emb(train_edge[1])

        # print(emb_start, emb_end)
        # pred = torch.dot(emb_start, emb_end)

        pred = torch.sum(emb_start * emb_end, dim=-1)
        pred = sigmoid(pred)
        loss = loss_fn(pred, train_label)

        dev_emb_start = emb(dev_edge[0])
        dev_emb_end = emb(dev_edge[1])
        dev_pred = torch.sum(dev_emb_start * dev_emb_end, dim=-1)
        dev_pred = F.sigmoid(dev_pred)
        dev_metric = F.mse_loss(dev_pred, dev_label)

        test_emb_start = emb(test_edge[0])
        test_emb_end = emb(test_edge[1])
        test_pred = torch.sum(test_emb_start * test_emb_end, dim=-1)
        test_pred = F.sigmoid(test_pred)
        test_metric = F.mse_loss(test_pred, test_label)

        if dev_metric < best_dev_metric - 1e-5:
            # better loss found
            patience_counter = 0
            best_dev_metric = dev_metric
            best_model_test_metric = test_metric
        else:
            patience_counter += 1

        if patience_counter > hpara.patience:
            break
        # if ep % 100 == 0:
        #     print(f"train loss:{loss}, test Loss: {best_model_test_metric}, ep:{ep}")
        loss.backward()
        optimizer.step()

    print(f"---------------------test Loss: {best_model_test_metric}, epoch:{ep}", )
    return best_model_test_metric


def random_and_split_data(_edge_list, _label, train_r, validate_r):
    """

    :param validate_r:
    :param train_r:
    :param _edge_list:
    :param _label:
    :return:
    """
    assert len(_edge_list) == len(_label)
    _ids = [i for i in range(len(_edge_list))]
    random.shuffle(_ids)
    rand_edge_list, rand_label = [], []
    for i in _ids:
        rand_edge_list.append(_edge_list[i])
        rand_label.append(_label[i])
    train_edge_list = rand_edge_list[0:int(train_r * len(rand_edge_list))]
    train_label = rand_label[0:int(train_r * len(rand_label))]
    validate_edge_list = rand_edge_list[
                         int(train_r * len(rand_edge_list)):int((train_r + validate_r) * len(rand_edge_list))]
    validate_label = rand_label[int(train_r * len(rand_label)):int((train_r + validate_r) * len(rand_label))]
    test_edge_list = rand_edge_list[int((train_r + validate_r) * len(rand_edge_list)):len(rand_edge_list)]
    test_label = rand_label[int((train_r + validate_r) * len(rand_label)):len(rand_label)]
    return rand_edge_list, rand_label, train_edge_list, train_label, validate_edge_list, validate_label, \
           test_edge_list, test_label


def main(args, G):
    results = {}
    for r_idx in range(len(args.train_ratios)):
        results[args.train_ratios[r_idx]] = [0, 0]
    for r_seed in args.seeds:
        random.seed(r_seed)
        # G is an undirected graph
        # print(type(G))
        # Visualize the graph
        # nx.draw(G, with_labels=True)
        # plt.show()

        num_edges = G.number_of_edges()
        num_nodes = G.number_of_nodes()
        # avg_degree = average_degree(num_edges, num_nodes)
        # print(f"Average degree of karate club network is {avg_degree}")

        # avg_cluster_coef = average_clustering_coefficient(G)
        # print("Average clustering coefficient of karate club network is {}".format(avg_cluster_coef))

        beta = 0.8
        r0 = 1 / G.number_of_nodes()
        node = 0

        # torch_test()
        # pos: exist in G, neg: in-exit in G
        pos_edge_list = graph_to_edge_list(G)

        # Sample 78 negative edges
        neg_edge_list = sample_negative_edges(G, 2)
        pos_edge_index = edge_list_to_tensor(pos_edge_list)
        # print("The pos_edge_index tensor has shape {}".format(pos_edge_index.shape))
        # print("The pos_edge_index tensor has sum value {}".format(torch.sum(pos_edge_index)))
        # Transform the negative edge list to tensor
        neg_edge_index = edge_list_to_tensor(neg_edge_list)
        # print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))

        # embedding_test()

        emb = create_node_emb(num_node=100, _embedding_dim=32)
        ids = torch.LongTensor([0, 3])
        # Print the embedding layer
        # print("Embedding: {}".format(emb))

        # An example that gets the embeddings for node 0 and 3
        # print(emb(ids))

        # visualize_emb(emb)

        # loss_fn = nn.BCELoss()
        loss_fn = nn.MSELoss()
        sigmoid = nn.Sigmoid()

        # print(pos_edge_index.shape)

        # Generate the positive and negative labels
        pos_label = torch.ones(pos_edge_index.shape[1], )
        neg_label = torch.zeros(neg_edge_index.shape[1], )

        jc_label = nx.jaccard_coefficient(G, G.edges)
        edge_list = []
        labels = []
        for u, v, p in jc_label:
            # print(print(f"({u}, {v}) -> {p:.8f}"))
            edge_list.append((u, v))
            labels.append(p)
        assert len(args.train_ratios) == len(args.dev_ratios)
        for r_idx in range(len(args.train_ratios)):
            rand_edge_list, rand_label, \
            train_edge_list, train_label, validate_edge_list, validate_label, test_edge_list, test_label = \
                random_and_split_data(edge_list, labels, train_r=args.train_ratios[r_idx],
                                      validate_r=args.dev_ratios[r_idx])
            train_edge = edge_list_to_tensor(train_edge_list)
            train_label = torch.tensor(train_label)
            validate_edge = edge_list_to_tensor(validate_edge_list)
            validate_label = torch.tensor(validate_label)
            test_edge = edge_list_to_tensor(test_edge_list)
            test_label = torch.tensor(test_label)
            # Concat positive and negative labels into one tensor
            # train_label = torch.cat([pos_label, neg_label], dim=0)

            # Concat positive and negative edges into one tensor
            # Since the network is very small, we do not split the edges into val/test sets
            # train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            # print(train_edge.shape)

            model_metric = \
                train(emb, args, loss_fn, sigmoid, train_label, train_edge, validate_label, validate_edge, test_label,
                      test_edge)
            results[args.train_ratios[r_idx]][0] += 1
            model_metric = float(model_metric)
            results[args.train_ratios[r_idx]][1] += (model_metric - results[args.train_ratios[r_idx]][1]) / \
                                                    results[args.train_ratios[r_idx]][0]
        print(results)

        json.dump(results, open("results.json", "w"))
    print(results)


if __name__ == '__main__':
    _args = get_args()
    graph = nx.dense_gnm_random_graph(100, 2000, seed=1)
    main(args=_args, G=graph)
