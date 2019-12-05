from math import exp
from random import seed
from random import random
# import utils 
import os 
import pdb 
DATA_DIR_PATH = './data'
NET_DIR_PATH = './nets'


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network
 
# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation
 
# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
 
# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']
 
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            print( row[-1] )
            pdb.set_trace()
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
 


def load_data(filename): 
    """Load data
    filename: string full filename of file to load (must be in DATA_DIR_PATH)
    returns int num_features, int num_outputs, list of lists dataset
    """
    # initialize list of lists of targest and inputs to be populated by data
    # inputs = [] 
    # targets = [] 
    dataset = []
    data_path = os.path.join( DATA_DIR_PATH, filename)
    with open( data_path, 'r') as f: 
        lines = f.readlines()
        num_examples, num_features, num_outputs = [ int(n) for n in lines[0].split(' ') ]
        for i in range( 1, 1+num_examples ): 
            values = [ float(n) for n in lines[i].split(' ') ]
            # inputs.append( values[:num_features] )
            # targets.append( values[-num_outputs:])
            dataset.append( values )
    # print(inputs)
    # print(targets)
    # print('INPUT SIZE: ', len(inputs))
    # print('TARGET SIZE: ', len(targets))
    return num_features, num_outputs, dataset


def load_neural_net(filename): 
    """Load neural net
    filename: string full filename of file to load (must be in NET_DIR_PATH)
    returns list of lists weights 
    """
    # weights = { 'w_input_hidden': [], 
    #             'w_hidden_output': []
    #           }

    weights = [[],[]] 
    data_path = os.path.join( NET_DIR_PATH, filename)
    with open( data_path, 'r') as f: 
        lines = f.readlines()
        num_input_nodes, num_hidden_nodes, num_output_nodes = [ int(n) for n in lines[0].split(' ') ]
        
        for i in range( 1, 1+num_hidden_nodes ): 
            values = [ float(n) for n in lines[i].split(' ') ]
            weights[0].append( values )

        for i in range( 1+num_hidden_nodes, 1+num_hidden_nodes+num_output_nodes ):
            values = [ float(n) for n in lines[i].split(' ') ]
            weights[1].append(values)

    # print(weights['w_input_hidden'])
    # print(weights['w_hidden_output'])
    # print('INPUT->HIDDEN SIZE: ', len(weights['w_input_hidden'][0]))
    # print('HIDDEN SIZE: ', len(weights['w_hidden_output'][0]))

    # print(weights[0])
    # print(weights[1])

    print('INPUT->HIDDEN WEIGHTS: (+1 FOR BIAS)', len(weights[0][0]))
    print('HIDDEN->OUTPUT WEIGHTS: (+1 FOR BIAS)', len(weights[1][0]))

    return num_input_nodes, num_hidden_nodes, num_output_nodes, weights

if __name__ == '__main__':
    # Test training backprop algorithm
    seed(1)
    # dataset = [[2.7810836,2.550537003,0],
    #     [1.465489372,2.362125076,0],
    #     [3.396561688,4.400293529,0],
    #     [1.38807019,1.850220317,0],
    #     [3.06407232,3.005305973,0],
    #     [7.627531214,2.759262235,1],
    #     [5.332441248,2.088626775,1],
    #     [6.922596716,1.77106367,1],
    #     [8.675418651,-0.242068655,1],
    #     [7.673756466,3.508563011,1]]

    data_filename = 'wdbc.TRAIN'
    network_filename = 'sample.NNWDBC.init'
    num_input_nodes, num_hidden_nodes, num_output_nodes, weights = load_neural_net(network_filename)
    num_features, num_outputs, dataset = load_data(data_filename) 


    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = initialize_network(num_input_nodes, num_hidden_nodes, num_output_nodes)
    train_network(network, dataset, 0.1, 100 , n_outputs)
    for layer in network:
        print(layer)