class Neuron:
    """
    Neuron class of NN
    inputs: List of incoming connections
    weights: List of weights to incoming connections
    """
    def __init__(self, activation, weights=None, inputs=None):
        self.weights = weights or []
        self.inputs = inputs or []
        self.value = None
        self.activation = activation

def network(input_units, num_hidden_nodes, output_units, activation):
    """
    Create Directed Acyclic Network of given number layers.
    hidden_layers_sizes : List number of neuron units in each hidden layer
    excluding input and output layers
    """
    layers_sizes = [input_units] + [num_hidden_nodes] + [output_units]
    net = [[Neuron(activation) for _ in range(size)] for size in layers_sizes]
    n_layers = len(net)

    # make connection
    for i in range(1, n_layers):
        for n in net[i]:
            # fully connected network 
            # (connect all neurons in layer to all neurons in previous layer)
            for k in net[i - 1]:
                n.inputs.append(k)
                n.weights.append(0)
    return net