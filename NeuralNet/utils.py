import os 
DATA_DIR_PATH = './data'
NET_DIR_PATH = './nets'


def load_data(filename): 
    """Load data
    filename: string full filename of file to load (must be in DATA_DIR_PATH)
    returns int num_features, int num_outputs, list of lists dataset
    """
    dataset = []
    data_path = os.path.join( DATA_DIR_PATH, filename)
    with open( data_path, 'r') as f: 
        lines = f.readlines()
        num_examples, num_features, num_outputs = [ int(n) for n in lines[0].split(' ') ]
        for i in range( 1, 1+num_examples ): 
            values = [ float(n) for n in lines[i].split(' ') ]
            dataset.append( values )
    return num_features, num_outputs, dataset


def load_neural_net(filename): 
    """Load neural net
    filename: string full filename of file to load (must be in NET_DIR_PATH)
    returns list of lists weights 
    """

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

    print('INPUT->HIDDEN WEIGHTS: (+1 FOR BIAS)', len(weights[0][0]))
    print('HIDDEN->OUTPUT WEIGHTS: (+1 FOR BIAS)', len(weights[1][0]))
    return num_input_nodes, num_hidden_nodes, num_output_nodes, weights

def save_neural_net(output_filename, num_input_nodes, num_hidden_nodes, num_output_nodes, weights): 
    data_path = os.path.join( NET_DIR_PATH, output_filename)
    with open( data_path, 'w' ) as f: 
        f.write(' '.join( [str(num_input_nodes), str(num_hidden_nodes), str(num_output_nodes)]  ) + os.linesep)
        for layers in weights:
            for layer in layers: 
                f.write(' '.join( str(x) for x in layer ) + os.linesep)
            

