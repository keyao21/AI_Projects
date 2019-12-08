from math import exp
from random import seed
from random import random
import utils 
import pdb 


def create_network(n_inputs, n_hidden, n_outputs, weights= None):
    """
    initialize neural network with intiial set of weights 
    (can be either trained or untrained)
    and number of inputs, hidden nodes, and outputs
    return network list of dicts ('weights' : list of lists of weights)
    """
    network = list()
    hidden_layer = [
        { 'weights': weights[0][i] }
        for i in range(n_hidden)
    ]
    network.append(hidden_layer)
    output_layer = [
        { 'weights': weights[1][i] }
        for i in range(n_outputs)
    ]
    network.append(output_layer)
    return network
 
def activate(weights, inputs):
    # Neuron is the sum product of weights and inputs, plus -1 * bias weight
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
 
def forward_propagate(network, row):
    # Forward propagation
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def loss_function( y_hat, y ):
    # error function for scalar values 
    return (y_hat  - y)

def train_network(network, train, l_rate, n_epoch, n_inputs, n_outputs):
    """
    Given a number of epochs, iterate through each training example and learn 
    the network weights through backpropagation
    """
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            # expected = [0 for i in range(n_outputs)]
            # print( row[-1] )
            # expected[int(row[-1])] = 1

            # pdb.set_trace()
            expected = row[-n_outputs:]
            sum_error += sum([(expected[i]-outputs[i]) for i in range(len(expected))])

            # backward propagate error
            for i in reversed(range(len(network))):
                layer = network[i]
                errors = list()
                if i != len(network)-1:
                    for j in range(len(layer)):
                        error = 0.0
                        for neuron in network[i + 1]:
                            error += (neuron['weights'][j] * neuron['delta'])
                            # error += loss_function(neuron['weights'][j], neuron['delta'])
                        errors.append(error)
                else:
                    for j in range(len(layer)):
                        neuron = layer[j]
                        # errors.append( expected[j] - neuron['output'] )
                        errors.append(  loss_function(expected[j], neuron['output']) )
                
                # print( errors )
                for j in range(len(layer)):
                    neuron = layer[j]
                    neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])

            # update weights
            for i in range(len(network)):
                inputs = row[:n_inputs]
                if i != 0:
                    inputs = [neuron['output'] for neuron in network[i - 1]]
                for neuron in network[i]:
                    for j in range(len(inputs)): 
                        neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                    neuron['weights'][-1] += l_rate * neuron['delta']
                    # pdb.set_trace()
        print('epoch :{0:d}, error: {1:.5f}'.format(epoch, sum_error)) 

def predict(network, row):
    # Make a prediction with a network
    outputs = forward_propagate(network, row)
    return outputs 


