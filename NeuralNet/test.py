from neuralnet import create_network, predict
import utils
import argparse 
from config import *
import pdb 

def test_neural_net( trained_network_filename, testing_data_filename, results_filename): 
    num_input_nodes, num_hidden_nodes, num_output_nodes, weights = utils.load_neural_net(trained_network_filename)
    num_features, num_outputs, test_dataset = utils.load_data(testing_data_filename) 
    network = create_network(num_input_nodes, num_hidden_nodes, num_output_nodes, weights)
    # confusion_matrices[output_idx][ predicted ][ expected ]
    confusion_matrices = [
            [
                [0 , 0],
                [0 , 0]
            ]
            for _ in range(num_outputs)
        ]

    # num_features, num_outputs, dataset = load_data(data_filename) 
    for row in test_dataset:    
        prediction = predict(network, row)[-num_outputs:]
        expected = row[-num_outputs:]
        print("=="*10)
        print( prediction )
        print( expected )

        for i in range(num_outputs): 
            confusion_matrices[i][ int(round( prediction[i] )) ][ int(expected[i]) ] += 1.0

    outputs_metrics = utils.calc_metrics( confusion_matrices )
    utils.save_metrics( results_filename, outputs_metrics )
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=['WDBC','Grades', 'Audit'], help='either WDBC, Grades, Audit')
    args = parser.parse_args()
    print( args.demo ) 

    if args.demo in CONFIGS.keys(): 
        test_neural_net( **CONFIGS[ args.demo ]['TEST'] )
    else: 
        trained_network_filename = input("Output network filename: ")
        testing_data_filename = input("Test data filename: ")
        results_filename = input("Results filename: ")
        test_neural_net( trained_network_filename, testing_data_filename, results_filename)
        


