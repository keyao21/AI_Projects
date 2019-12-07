from math import exp
from random import seed
from random import random
import utils 
import pdb 


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs, weights= None):
    network = list()
    # hidden_layer = [{'weights':[round( random(), 3) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    hidden_layer = [
        { 'weights': weights[0][i] }
        for i in range(n_hidden)
    ]
    # pdb.set_trace() 
    network.append(hidden_layer)
    # output_layer = [{'weights':[round( random(),3 ) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    output_layer = [
        { 'weights': weights[1][i] }
        for i in range(n_outputs)
    ]


    for i in range(n_hidden): 
        print( ' '.join(str(x) for x in hidden_layer[i]['weights']) ) 

    for i in range(n_outputs): 
        print( ' '.join(str(x) for x in output_layer[i]['weights']) ) 

    # print( hidden_layer)
    # print( output_layer)
    network.append(output_layer)
    return network
 
# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
        # print( activation )
    return activation
 
# Transfer neuron activation
def transfer(activation):
    # print( 1.0 / (1.0 + exp(-activation)) )
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
 
def loss_function( y_hat, y ):
    # error function for scalar values 
    return (y_hat  - y)

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_inputs, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            # print( row[-1] )
            # expected[int(row[-1])] = 1

            expected = row[-n_outputs:]
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            # pdb.set_trace()

            # backward_propagate_error
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
                    neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

            # update weights
            for i in range(len(network)):
                inputs = row[:n_inputs]
                if i != 0:
                    inputs = [neuron['output'] for neuron in network[i - 1]]
                for neuron in network[i]:
                    for j in range(len(inputs)):
                        neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                        # pdb.set_trace()
                    neuron['weights'][-1] += l_rate * neuron['delta']

        print('>epoch={0:d}, error={1:.2f}'.format(epoch, sum_error)) 

# Make a prediction with a network
def predict(network, row, num_outputs):
    outputs = forward_propagate(network, row)
    # pdb.set_trace()
    if num_outputs == 1: 
        return outputs[0] 
    return outputs.index(max(outputs))


