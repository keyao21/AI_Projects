from neuralnet import train_network, create_network
import utils 
import argparse 
from config import *

def train_neural_net(learning_rate, num_epoch, training_data_filename, init_network_filename, trained_network_filename): 
    
    num_input_nodes, num_hidden_nodes, num_output_nodes, weights = utils.load_neural_net(init_network_filename)
    num_features, num_outputs, dataset = utils.load_data(training_data_filename) 

    network = create_network(num_hidden_nodes, num_output_nodes, weights)
    train_network(network, dataset, learning_rate, num_epoch , num_features, num_outputs)
    for layer in network:
        print(layer)

    utils.save_neural_net( trained_network_filename, num_input_nodes, num_hidden_nodes, num_output_nodes, weights )
    return network 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=['WDBC', 'WDBC_mini', 'Grades', 'Audit'], help='either WDBC, Grades, Audit')
    args = parser.parse_args()
    print( args.demo ) 

    if args.demo in CONFIGS.keys(): 
        train_neural_net( **CONFIGS[ args.demo ]['TRAIN'] )
    else: 
        init_network_filename = str(input("Initial network filename: "))
        training_data_filename = str(input("Training data filename: "))
        trained_network_filename = str(input("Output network filename: "))
        learning_rate = float(input("Learning rate: "))
        num_epoch = int(input("Number of Epochs: "))
        train_neural_net( learning_rate, num_epoch, training_data_filename, init_network_filename, trained_network_filename )



