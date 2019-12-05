import math 
import os 
import operator 

DATA_DIR_PATH = './data'
NET_DIR_PATH = './nets'

def scalar_vector_product(x, y):
    """Return vector as a product of a scalar and a vector"""
    return [ x*_y for _y in y]

def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))

def dot_product(x, y):
    """Return the sum of the element-wise product of vectors x and y."""
    return sum(_x * _y for _x, _y in zip(x, y))

def sigmoid_derivative(value):
    return value * (1 - value)

def sigmoid(x):
    """Return activation value of x with sigmoid function."""
    return 1 / (1 + math.exp(-x))

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


def load_data(filename): 
    """Load data
    filename: string full filename of file to load (must be in DATA_DIR_PATH)
    returns int num_features, int num_outputs, list of lists inputs, list of lists targets
    """
    # initialize list of lists of targest and inputs to be populated by data
    inputs = [] 
    targets = [] 

    data_path = os.path.join( DATA_DIR_PATH, filename)
    with open( data_path, 'r') as f: 
        lines = f.readlines()
        num_examples, num_features, num_outputs = [ int(n) for n in lines[0].split(' ') ]
        for i in range( 1, 1+num_examples ): 
            values = [ float(n) for n in lines[i].split(' ') ]
            inputs.append( values[:num_features] )
            targets.append( values[-num_outputs:])
    
    # print(inputs)
    # print(targets)
    print('INPUT SIZE: ', len(inputs))
    print('TARGET SIZE: ', len(targets))
    return num_features, num_outputs, inputs, targets


if __name__ == '__main__':
    # load_data('wdbc.TRAIN')
    load_neural_net('sample.NNWDBC.init')