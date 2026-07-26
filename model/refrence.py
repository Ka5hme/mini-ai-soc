# from dense_layer import dense_layer
# from network import network
#
# """
# Golden reference model for the mini AI SoC project.
#
# Defines a small 8 -> 4 -> 2 neural network and executes inference
# using the integer dense-layer implementation.
#
# Expected final output:
#     [75, 25]
# """
#
# inputs  = [2, 4, 1, 3, 5, 2, 6, 1]
# weights_layer1 = [
#     [ 3, -1,  2,  4, -2,  1,  2, -3],  # Neuron 0
#     [-2,  3,  1, -1,  2,  4, -2,  1],  # Neuron 1
#     [ 1,  2, -3,  2,  1, -2,  3,  2],  # Neuron 2
#     [ 2, -2,  4,  1, -1,  3,  1, -2]   # Neuron 3
# ]
# bias_layer1 = [-30, 5, -10, 8]
#
# weights_layer2 = [
#     [2, -1, 3, 1],
#     [-1, 2, 1, -2]
# ]
#
# bias_layer2 = [5, -3]
#
# layers = [
#     {
#         "weights": weights_layer1,
#         "bias": bias_layer1
#     },
#
#     {
#         "weights": weights_layer2,
#         "bias": bias_layer2
#     }
# ]
#
#
# # Run network
#
# output = network(inputs, layers)
#
# print("Network output:", output)
#
#
#
#
