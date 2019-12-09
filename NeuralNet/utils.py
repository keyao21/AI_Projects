from random import seed
from random import random
import os 
DATA_DIR_PATH = './data'
NET_DIR_PATH = './nets'
RESULTS_DIR_PATH = './results'

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

            _bias_weight = values[0]
            _weights = values[1:]
            values_bias_to_end = _weights + [_bias_weight]

            weights[0].append( values_bias_to_end )

        for i in range( 1+num_hidden_nodes, 1+num_hidden_nodes+num_output_nodes ):
            values = [ float(n) for n in lines[i].split(' ') ]

            _bias_weight = values[0]
            _weights = values[1:]
            values_bias_to_end = _weights + [_bias_weight]

            weights[1].append(values_bias_to_end)

    print('INPUT->HIDDEN WEIGHTS: (+1 FOR BIAS)', len(weights[0][0]))
    print('HIDDEN->OUTPUT WEIGHTS: (+1 FOR BIAS)', len(weights[1][0]))
    return num_input_nodes, num_hidden_nodes, num_output_nodes, weights

def save_neural_net(output_filename, num_input_nodes, num_hidden_nodes, num_output_nodes, weights): 
    """
    Save down neural net specs to output file with set specs
    3 digit double precision 
    """
    data_path = os.path.join( NET_DIR_PATH, output_filename)
    with open( data_path, 'w' ) as f: 
        f.write(' '.join( [str(num_input_nodes), str(num_hidden_nodes), str(num_output_nodes)]  ) + os.linesep)
        for layers in weights:
            for layer in layers: 
                _bias_weight = layer[-1]
                _weights = layer[:-1]
                values_bias_to_end = [_bias_weight] + _weights 
                f.write(' '.join( "{0:.3f}".format(round(x, 3))  for x in values_bias_to_end ) + os.linesep)
            

def calc_accuracy(A, B, C, D): 
    try: 
        overall_accuracy = (A + D) / (A + B + C + D)
    except: overall_accuracy = 0.0
    return overall_accuracy

def calc_precision(A, B, C, D): 

    try: precision = A / (A + B)
    except: precision = 0.0 
    return precision 

def calc_recall(A, B, C, D): 
    try: recall = A / (A + C)
    except: recall = 0.0
    return recall 

def calc_f1(A, B, C, D): 
    try: f1 = (2*calc_precision(A,B,C,D)*calc_recall(A,B,C,D)) / (calc_precision(A,B,C,D)+calc_recall(A,B,C,D))
    except: f1 = 0.0
    return f1 


def calc_metrics( confusion_matrices ):
    """
    Reports output metrics for individual output types, 
    as well as micro and macro averaged output metrics
    Returns list of lists outputs_metrics
    """

    outputs_metrics = []

    macro_overall_accuracy = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    micro_sum_A = 0.0
    micro_sum_B = 0.0
    micro_sum_C = 0.0
    micro_sum_D = 0.0
    
    print("\nCalculating individual output metrics...")
    for output_idx, confusion_matrix in enumerate( confusion_matrices ):

        # Note that A is calculated based on index 1,1 (not 0,0) in the confusion matrix
        A = confusion_matrix[1][1]
        B = confusion_matrix[1][0]
        C = confusion_matrix[0][1]
        D = confusion_matrix[0][0]
        
        overall_accuracy = calc_accuracy( A, B, C, D)
        overall_precision = calc_precision( A, B, C, D)
        overall_recall = calc_recall( A, B, C, D)
        overall_f1 = calc_f1( A, B, C, D )
        
        ### performing precalculations for micro and macro level metrics ###
        micro_sum_A += A
        micro_sum_B += B
        micro_sum_C += C 
        micro_sum_D += D

        macro_overall_accuracy  += overall_accuracy 
        macro_precision += overall_precision
        macro_recall += overall_recall 
        macro_f1 += overall_f1 
        ##############################################################3#####

        print("="*30)
        print("Confusion matrix for output {0}".format(output_idx))
        print( confusion_matrix )
        print("A:", A )
        print("B:", B )
        print("C:", C )
        print("D:", D )
        print('overall_accuracy:', overall_accuracy)
        print('precision:', overall_precision)
        print('recall:', overall_recall)
        print('f1:', overall_f1)
        outputs_metrics.append( [A, B, C, D, overall_accuracy, overall_precision, overall_recall, overall_f1] )


    print("\nCalculating macro output metrics... ")
    macro_overall_accuracy *= (1.0 / len(confusion_matrices) )
    macro_precision *= (1.0 / len(confusion_matrices) )
    macro_recall *= (1.0 / len(confusion_matrices) )
    macro_f1 *= (1.0 / len(confusion_matrices) )
    print("="*30)
    print("macro overall accuracy:", macro_overall_accuracy)
    print("macro precision:", macro_precision)
    print("macro recall:", macro_recall)
    print("macro f1:", macro_f1)
    outputs_metrics.append( [macro_overall_accuracy, macro_precision, macro_recall, macro_f1] )

    print("\nCalculating micro output metrics... ")
    micro_overall_accuracy = calc_accuracy( micro_sum_A, micro_sum_B, micro_sum_C, micro_sum_D )
    micro_precision = calc_precision( micro_sum_A, micro_sum_B, micro_sum_C, micro_sum_D )
    micro_recall = calc_recall( micro_sum_A, micro_sum_B, micro_sum_C, micro_sum_D )
    micro_f1 = calc_f1( micro_sum_A, micro_sum_B, micro_sum_C, micro_sum_D )
    
    print("="*30)
    print("A:", micro_sum_A )
    print("B:", micro_sum_B )
    print("C:", micro_sum_C )
    print("D:", micro_sum_D )
    print("micro_overall_accuracy:", micro_overall_accuracy )
    print("micro_precision:", micro_precision )
    print("micro_recall:", micro_recall )
    print("micro_f1:", micro_f1 )
    outputs_metrics.append( [micro_overall_accuracy, micro_precision, micro_recall, micro_f1] )

    return outputs_metrics


def save_metrics( output_filename, outputs_metrics ): 
    output_path = os.path.join( RESULTS_DIR_PATH, output_filename)
    with open( output_path, 'w' ) as f:
        for line in outputs_metrics[:-2]:
            f.write( ' '.join(  ["{0:d}".format(int(x)) for x in line[:4] ] \
                                + ["{0:.3f}".format(round(x, 3)) for x in line[4:]]) \
                                + os.linesep )
        for line in outputs_metrics[-2:]: 
            f.write(' '.join( "{0:.3f}".format(round(x, 3)) for x in line ) + os.linesep)


def create_network_file( network_filename, num_input_nodes, num_hidden_nodes, num_output_nodes ): 
    output_path = os.path.join( NET_DIR_PATH, network_filename)
    with open( output_path, 'w' ) as f: 
        f.write(' '.join( [str(num_input_nodes), str(num_hidden_nodes), str(num_output_nodes)]  ) + os.linesep)
        
        for _ in range(num_hidden_nodes): 
            f.write(' '.join( "{0:.3f}".format(  round(random(), 3)  )  for _ in range(num_input_nodes) ) + os.linesep)
        
        for _ in range(num_output_nodes): 
            f.write(' '.join( "{0:.3f}".format(  round(random(), 3)  )  for _ in range(num_hidden_nodes) ) + os.linesep)
            
if __name__ == '__main__':
    num_input_nodes = 26
    num_hidden_nodes = 10
    num_output_nodes = 1
    network_filename = "kevin.NNAudit.init"
    create_network_file( network_filename, num_input_nodes, num_hidden_nodes, num_output_nodes )