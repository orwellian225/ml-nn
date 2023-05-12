import numpy as np


def init_layer(layer_node_count, next_node_count):
    weight_matrix = []
    for i in range(next_node_count):
        row = []
        # + 1 because its the bias vector in that column
        for j in range(layer_node_count + 1):
            row.append(np.random.rand())
        weight_matrix.append(np.array(row))

    return np.array(weight_matrix)


def init_network(layer_counts):
    layers = []
    for i in range(len(layer_counts) - 1):
        layers.append(init_layer(layer_counts[i], layer_counts[i + 1]))

    return np.array(layers)


# Returns the output vector of the network until the last layer of the provided
# slice
def fprop_layer(network_slice, input, activation_f):
    for layer in network_slice:
        input = np.append(input, 1)

        output = np.matmul(layer, input)
        for i in range(len(output)):
            output[i] = activation_f(output[i])
        input = output

    return output


# Returns the matrix of activation values for the network
# last layer are the results of the network
def fprop_network(network, input, activation_f):
    result = []
    for layer in network:
        input = np.append(input, 1)

        output = np.matmul(layer, input)
        for i in range(len(output)):
            output[i] = activation_f(output[i])
        result.append(output)
        input = output

    return result


# Returns the result of the neural network evaluation
def eval_network(network, input, activation_f):
    return fprop_network(network, input, activation_f)[-1]


# Returns the small_delta (error) at the first layer of the provided network
# Therfore to calculate the error up until a specific layer, input a network
# slice of every layer after (and including) the layer you want the error for
def bprop_network(network, input, correct_class, activation_f):
    layer_activations = fprop_network(network, input, activation_f)

    layer_error = layer_activations[-1] - correct_class
    layer_errors = [layer_error]  # add the last error layer to the matrix

    for i in (len(network) - 2, 0, -1):
        layer_activation = layer_activations[i]
        layer_error = np.matmul(network[i].transpose(), layer_error) * layer_activation * (1 - layer_activation)
        layer_errors.append(layer_error)
    layer_errors = np.array(layer_errors)

    gradients = np.zeros((len(network) - 1, max(len(*network[:]))))
    for l in range(len(layer_activations)):
        for i in range(len(layer_activations[l])):
            for j in range(len(layer_errors[l])):
                gradients[i][j] += layer_activations[l][i] * layer_errors[l][j]

    for i in range(len(gradients)):
        for j in range(len(gradients[i])):
            if j == len(gradients[i]) - 1:
                gradients[i][j] = 1 / len(gradients[i]) * gradients[i][j]
            else:
                # Regularization goes here
                gradients[i][j] = 1 / len(gradients[i]) * gradients[i][j]

    return gradients
