from math import exp
import utils 
import pdb 


def create_network(num_hidden, num_outputs, weights ):
    """
    initialize neural network with intiial set of weights 
    (can be either trained or untrained)
    and number of inputs, hidden nodes, and outputs
    return network list of dicts ('weights' : list of lists of weights)
    """
    network = list()
    hidden_layer = [
        { 'weights': weights[0][i] }
        for i in range(num_hidden)
    ]
    network.append(hidden_layer)
    output_layer = [
        { 'weights': weights[1][i] }
        for i in range(num_outputs)
    ]
    network.append(output_layer)
    return network
 
def activate(weights, inputs):
    # node input to activation function is the 
    # sum product of weights and inputs, plus -1 * bias weight
    activation = weights[-1] *(-1)
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation
 
def sigmoid(activation):
    # Sigmoid activation function
    return 1.0 / (1.0 + exp(-activation))
 
def sigmoid_derivative(output):
    # Derivative of sigmoid 
    # output is from sigmoid function
    return output * (1.0 - output)
 
def forward_propagate(network, example):
    # Forward propagation
    inputs = example
    for layer in network:
        new_inputs = []
        for node in layer:
            activation = activate(node['weights'], inputs)
            node['output'] = sigmoid(activation)
            new_inputs.append(node['output'])
        inputs = new_inputs
    return inputs

def loss_function( y_hat, y ):
    # error function for scalar values 
    return (y_hat  - y)

def train_network(network, dataset, learning_rate, num_epoch, num_inputs, num_outputs):
    """
    Given a number of epochs, iterate through each training example and learn 
    the network weights through backpropagation
    """
    for epoch in range(num_epoch):
        sum_error = 0
        for example in dataset:
            outputs = forward_propagate(network, example)
            expected = example[-num_outputs:]
            sum_error += sum([(expected[i]-outputs[i]) for i in range(len(expected))])
            # backward propagate error
            for i in reversed(range(len(network))):
                layer = network[i]
                errors = [] # intialize list fo errors (one for each node in layer)
                if i != len(network)-1:
                    for j, _ in enumerate(layer):
                        error = 0.0
                        for node in network[i+1]:
                            error += node['weights'][j] * node['delta']
                        errors.append(error)
                else:
                    for j, node in enumerate(layer): 
                        errors.append(  loss_function(expected[j], node['output']) )
                for j, node in enumerate(layer):
                    node['delta'] = errors[j] * sigmoid_derivative(node['output'])
            # update weights
            for i in range(len(network)):
                inputs = example[:num_inputs]
                if i != 0:
                    inputs = [node['output'] for node in network[i - 1]]
                for node in network[i]:
                    for j in range(len(inputs)): 
                        node['weights'][j] += learning_rate * node['delta'] * inputs[j]
                    node['weights'][-1] += learning_rate * node['delta'] * (-1)
        print('epoch :{0:d}, error: {1:.5f}'.format(epoch, sum_error)) 

def predict(network, example):
    # Prediction function given network and inputs
    outputs = forward_propagate(network, example)
    return outputs 


