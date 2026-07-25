from dense_layer import dense_layer

def network(inputs, layers):

    """
    Args:
        inputs: Initial network input.
        layers: List of layer configurations containing weights and biases.
    Returns:
        Output of the final layer.
    """

    current_data = inputs
    for layer in layers:
        current_data = dense_layer(current_data, layer["weights"], layer["bias"])

    return current_data