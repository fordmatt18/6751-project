import torch
import numpy as np
from abstract_environment import AbstractEnvironment
from sklearn.preprocessing import PolynomialFeatures

# use same w, parameters from which y is generated, every time without effecting other randomness in pytorch
state = torch.random.get_rng_state()
torch.manual_seed(10)
w = torch.randn(40,31)
torch.random.set_rng_state(state)

# polynomial transformation of context variables used to generate y
poly = PolynomialFeatures(degree=5, include_bias=False, interaction_only = True)

# calculate constraint matrix A which encodes flow constraints
node_coords = {}
node_labels = {}
counter = 0
for i in range(5):
    for j in range(5):
        node_labels[i,j] = counter
        node_coords[counter] = i,j
        counter += 1
edge_labels = {}
edge_nodes = {}
counter = 0
for coord in node_coords.values():
    if coord[0] < 4:
        edge_labels[node_labels[coord], node_labels[coord[0] + 1, coord[1]]] = counter
        edge_nodes[counter] = node_labels[coord], node_labels[coord[0] + 1, coord[1]]
        counter += 1
    if coord[1] < 4:
        edge_labels[ node_labels[coord], node_labels[coord[0], coord[1] + 1]] = counter
        edge_nodes[counter] = node_labels[coord], node_labels[coord[0], coord[1] + 1]
        counter += 1
A = np.zeros((25,40))
for node in node_labels.values():
    if (node_coords[node][0] + 1, node_coords[node][1]) in node_labels:
        A[node, edge_labels[node, node_labels[node_coords[node][0] + 1, node_coords[node][1]]]] = 1
    if (node_coords[node][0], node_coords[node][1] + 1) in node_labels:
        A[node, edge_labels[node, node_labels[node_coords[node][0], node_coords[node][1] + 1]]] = 1
    if (node_coords[node][0] - 1, node_coords[node][1]) in node_labels:
        A[node, edge_labels[node_labels[node_coords[node][0] - 1, node_coords[node][1]], node]] = -1
    if (node_coords[node][0], node_coords[node][1] - 1) in node_labels:
        A[node, edge_labels[node_labels[node_coords[node][0], node_coords[node][1] - 1], node]] = -1 

class ShortestPathEnvironment(AbstractEnvironment):
    def __init__(self):
        AbstractEnvironment.__init__(self)
        self.context_dim = 5
        self.decision_dim = 40
        self.w = w
        self.poly = poly

    def sample_data(self, n):
        """
        samples n random data points from environment, each of which is a tuple
            containing context and corresponding cost vector
        :return: tuple (x, y), where x is of shape (n, context_dim),
            and y is of shape (n, decision_dim)
        """
        x = torch.randn(n, self.context_dim)
        poly_x = self.poly.fit_transform(x)
        noise = torch.tensor([3/4.0, 5/4.0], dtype=float)[torch.randint(2, (n,))]
        y = torch.diag(noise) @ ((w @ poly_x.T).T + 3) 
        return (x, y)
    
    def get_constraints(self):
        """
        :return: constraints for given environment
        """
        b = torch.zeros(25)
        b[0] = 1
        b[-1] = -1
        return {"A": A, "b": b}

    def get_context_dim(self):
        """
        :return: context_dim for given environment
        """
        return self.context_dim

    def get_decision_dim(self):
        """
        :return: decision_dim for given environment
        """
        return self.decision_dim
