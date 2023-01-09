import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub

from torch.nn import Linear
from torch_geometric.nn import GCNConv


# Visualization function for NX graph or PyTorch tensor
def visualize(h, color, save_fig_file=None, epoch=None, loss=None, accuracy=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                        f'Training Accuracy: {accuracy["train"] * 100:.2f}% \n'
                        f' Validation Accuracy: {accuracy["val"] * 100:.2f}%'),
                       fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    # plt.show()
    if save_fig_file:
        plt.savefig(save_fig_file)
    else:
        plt.show()


dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {(2 * data.num_edges) / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
# print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
# print(f'Contains self-loops: {data.has_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')

# print(data.edge_index.T)
print(data)

from IPython.display import Javascript  # Restrict height of output cell.

# display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

edge_index = data.edge_index
# print(edge_index.t())

from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)


# visualize(G, color=data.y)


class GCN(torch.nn.Module):
    def __init__(self, hidden_dim=4, _num_layers=3):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.num_layers = _num_layers

        self.convs = torch.nn.ModuleList()

        self.convs.append(GCNConv(dataset.num_features, hidden_dim))
        for l in range(self.num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, 2))
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, _edge_index):
        print(x.size(), _edge_index.size())
        h = self.convs[0](x, _edge_index)
        for l_num in range(1, self.num_layers):
            h = self.convs[l_num](h, _edge_index)
            # h = torch.nn.Dropout(h)
            h = h.tanh()

        # h = self.relu(h)
        # h = torch.nn.ReLU(h)

        # h = self.conv3(h, _edge_index)
        embeddings = h  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(embeddings)

        return out, embeddings


# model = GCN()
# print(model)
# _, h = model(data.x, data.edge_index)
# print(f'Embedding shape: {list(h.shape)}')

# visualize(h, color=data.y, save_fig_file='./lab0-random-gnn.png')

import time

model = GCN()
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.


def train(_data):
    optimizer.zero_grad()  # Clear gradients.
    out, _h = model(_data.x, _data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[_data.train_mask],
                     _data.y[_data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    accuracy = {}
    # Calculate training accuracy on our four examples
    predicted_classes = torch.argmax(out[_data.train_mask])  # [0.6, 0.2, 0.7, 0.1] -> 2
    target_classes = _data.y[_data.train_mask]
    accuracy['train'] = torch.mean(
        torch.where(predicted_classes==target_classes, 1, 0).float())

    # Calculate validation accuracy on the whole graph
    predicted_classes = torch.argmax(out)
    target_classes = _data.y
    accuracy['val'] = torch.mean(
        torch.where(predicted_classes==target_classes, 1, 0).float())

    return loss, _h, accuracy


for epoch in range(500):
    loss, h, accuracy = train(data)
    # Visualize the node embeddings every 10 epochs
    if epoch % 100 == 0:
        visualize(h, color=data.y, save_fig_file=f'./{epoch}_step_training_result.png', epoch=epoch, loss=loss, accuracy=accuracy)
        time.sleep(0.3)
