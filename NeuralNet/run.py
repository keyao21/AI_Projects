from utils import scalar_vector_product, vector_add, dot_product, sigmoid, sigmoid_derivative, load_data, load_neural_net
from neural_net import network
import time 
import pdb 


def random_weights(min_value, max_value, num_weights):
    import random 
    return [random.uniform(min_value, max_value) for _ in range(num_weights)]


def BackPropagationLearner(inputs, weights, targets, net, learning_rate, epochs, activation=sigmoid):
    # initialize weights as passed in
    # we only have weights for the hidden and output layers 
    for layer_idx, layer in enumerate(net[1:]):
        for node_idx, node in enumerate(layer):
            node.weights = random_weights(min_value=0.0, max_value=1.0, num_weights=len(node.weights))
            # node.weights = weights[layer_idx][node_idx]
    # CHECK:
    # net[1][0].weights should match the first weights line in the neural network file

    '''
    As of now dataset.target gives an int instead of list,
    Changing dataset class will have effect on all the learners.
    Will be taken care of later.
    '''
    o_nodes = net[-1]
    i_nodes = net[0]
    o_units = len(o_nodes)
    n_layers = len(net)

    for epoch in range(epochs):
        # Iterate over each example
        for e in range(len(inputs)):
            i_val = inputs[e]
            t_val = targets[e]

            # Activate input layer
            for v, n in zip(i_val, i_nodes):
                n.value = v

            # pdb.set_trace()

            # Forward pass
            for layer in net[1:]:
                for node in layer:
                    inc = [n.value for n in node.inputs]
                    in_val = dot_product(inc, node.weights)
                    node.value = node.activation(in_val)

            # Initialize delta
            delta = [[] for _ in range(n_layers)]

            # Compute outer layer delta
            # Error for the MSE cost function
            err = [t_val[i] - o_nodes[i].value for i in range(o_units)]
            
            delta[-1] = [sigmoid_derivative(o_nodes[i].value) * err[i] for i in range(o_units)]
            
            # Backward pass
            h_layers = n_layers - 2
            for i in range(h_layers, 0 , -1):
                layer = net[i]
                h_units = len(layer)
                nx_layer = net[i+1]
                # weights from each ith layer node to each i + 1th layer node
                w = [[node.weights[k] for node in nx_layer] for k in range(h_units)]
                delta[i] = [sigmoid_derivative(layer[j].value) * dot_product(w[j], delta[i+1]) for j in range(h_units)]

            #  Update weights
            for i in range(1, n_layers):
                layer = net[i]
                inc = [node.value for node in net[i-1]]
                units = len(layer)
                for j in range(units):
                    layer[j].weights = vector_add(layer[j].weights, scalar_vector_product(learning_rate * delta[i][j], inc))
                    
    return net


def NeuralNetLearner(data_filename, network_filename, learning_rate=0.01, epochs=100, activation = sigmoid):
    """
    data_filename: String name of file in data directory
    network_filename: String name of file in net directory

    learning_rate: Learning rate of gradient descent
    epochs: Number of passes over the dataset
    """

    print("Loading files...")
    num_input_nodes, num_hidden_nodes, num_output_nodes, weights = load_neural_net(network_filename)
    num_features, num_outputs, inputs, targets = load_data(data_filename) 
    i_units = num_features
    o_units = num_outputs

    # reconcile data and network file specs 
    if i_units != num_input_nodes: 
        print("num_input_nodes of network should match with num_features in dataset!")
        return 
    if o_units != num_output_nodes: 
        print("num_output_nodes of network should match with num_outputs in dataset!")
        return 

    # construct a network
    print("Intializing network...")
    raw_net = network(i_units, num_hidden_nodes, o_units, activation)
    print("Training network...")
    st = time.time()
    learned_net = BackPropagationLearner(inputs, weights, targets, raw_net, learning_rate, epochs, activation)
    print("Model finished (trained in {0:.2f} seconds)".format(time.time() - st))

    def predict(example):
        # Input nodes
        i_nodes = learned_net[0]
        # Activate input layer
        for v, n in zip(example, i_nodes):
            n.value = v
        # Forward pass
        for layer in learned_net[1:]:
            for node in layer:
                inc = [n.value for n in node.inputs]
                in_val = dot_product(inc, node.weights)
                node.value = node.activation(in_val)
        # Hypothesis
        # pdb.set_trace()
        o_nodes = learned_net[-1]

        if len(o_nodes) == 1: 
            return o_nodes[0].value
        else:
            return o_nodes.index(max(o_nodes, key=lambda node: node.value))
    return predict 

def test(trained_nn, test_data_filename): 
    num_features, num_outputs, inputs, targets = load_data(test_data_filename) 

    # confusion matrix 
    # confusion_matrix[ predicted ][ expected ]
    confusion_matrix = [
        [0 , 0],
        [0 , 0]
    ]

    for idx, example in enumerate(inputs): 
        expected = targets[idx][0]
        predicted = round( trained_nn(example) ) # 0 or 1
        print(expected, predicted)
        confusion_matrix[ int(predicted) ][ int(expected) ] += 1

    overall_accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0]+ confusion_matrix[1][1])
    precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    f1 = (2*precision*recall) / (precision+recall)

    print('overall_accuracy:', overall_accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

    pdb.set_trace()



if __name__ == '__main__':
    data_filename = 'wdbc.TRAIN'
    net_filename = 'sample.NNWDBC.init'
    trained_nn = NeuralNetLearner(  data_filename=data_filename, 
                                    network_filename=net_filename,
                                    learning_rate=0.01, 
                                    epochs=100, 
                                    activation = sigmoid
                                )
    test_data_filename = 'wdbc.TEST'
    test(trained_nn, test_data_filename )

