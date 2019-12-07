from neuralnet import initialize_network, predict
import utils
import argparse 
from config import *


def test_neural_net( trained_network_filename, testing_data_filename): 
    num_input_nodes, num_hidden_nodes, num_output_nodes, weights = utils.load_neural_net(trained_network_filename)
    num_features, num_outputs, test_dataset = utils.load_data(testing_data_filename) 
    network = initialize_network(num_input_nodes, num_hidden_nodes, num_output_nodes, weights)
    # confusion_matrix[ predicted ][ expected ]
    confusion_matrix = [
        [0 , 0],
        [0 , 0]
    ]

    # num_features, num_outputs, dataset = load_data(data_filename) 
    for row in test_dataset:
        prediction = predict(network, row, num_outputs)
        expected = row[-1]
        confusion_matrix[ int(round( prediction )) ][ int(expected) ] += 1.0

    try: overall_accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0]+ confusion_matrix[1][1])
    except: overall_accuracy = None
    
    try: precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    except: precision = None 
    
    try: recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    except: recall = None
    
    try: f1 = (2*precision*recall) / (precision+recall)
    except: f1 = None
    
    print("Confusion matrix: ")
    print( confusion_matrix)
    print('overall_accuracy:', overall_accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=['WDBC','grades'], help='either WDBC or Grades')
    args = parser.parse_args()
    print( args.demo ) 

    if args.demo in CONFIGS.keys(): 
        test_neural_net( **CONFIGS[ args.demo ]['TEST'] )
    else: 
        trained_network_filename = input("Output network filename: ")
        testing_data_filename = input("Test data filename: ")
        test_neural_net( trained_network_filename, testing_data_filename)
        
