from neuralnet import train_network, initialize_network
import utils 
import argparse 
from config import *

def train_neural_net(l_rate, n_epoch, training_data_filename, init_network_filename, trained_network_filename): 
    # Test training backprop algorithm
    # seed(1)
    
    num_input_nodes, num_hidden_nodes, num_output_nodes, weights = utils.load_neural_net(init_network_filename)
    num_features, num_outputs, dataset = utils.load_data(training_data_filename) 

    n_inputs = len(dataset[0]) - num_outputs
    n_outputs = 1
    network = initialize_network(num_input_nodes, num_hidden_nodes, num_output_nodes, weights)
    train_network(network, dataset, l_rate, n_epoch , n_inputs, n_outputs)
    for layer in network:
        print(layer)

    utils.save_neural_net( trained_network_filename, num_input_nodes, num_hidden_nodes, num_output_nodes, weights )
    return network 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=['WDBC','grades'], help='either WDBC or Grades')
    args = parser.parse_args()
    print( args.demo ) 

    if args.demo in CONFIGS.keys(): 
        train_neural_net( **CONFIGS[ args.demo ]['TRAIN'] )
    else: 
        init_network_filename = input("Initial network filename: ")
        training_data_filename = input("Training data filename: ")
        trained_network_filename = input("Output network filename: ")
        l_rate = int(input("Learning rate: "))
        n_epoch = int(input("Number of Epochs: "))
        train_neural_net( l_rate, n_epoch, training_data_filename, init_network_filename, trained_network_filename )



