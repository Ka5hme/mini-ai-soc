def dense_layer(inputs, weights, bias):
    """
    Compute one neural-network layer.
    Args:
        inputs: Input activations for the layer.
        weights: 2D weight matrix [neuron][input].
        bias: One bias value per neuron.
    Returns:
        List containing the output activation of each neuron.
    """

    outputs = []
    # Check input/weight lengths
    if len(bias) != len(weights):
        raise ValueError("Bias and weights must have the same length")

    for i in range(len(weights)):

        # Check input/weight lengths
        if len(inputs) != len(weights[i]):
            raise ValueError("Each neuron must have one weight per input")

        # Check bias is INT32
        if not (-2 ** 31 <= bias[i] <= 2 ** 31 - 1):
            raise ValueError("Bias outside INT32 range")

        ACC = 0

        for j in range(len(inputs)):

            # Check input and weight are INT8
            if not (-128 <= inputs[j] <= 127):
                raise ValueError("Input outside INT8 range")

            if not (-128 <= weights[i][j] <= 127):
                raise ValueError("Weight outside INT8 range")

            product = inputs[j] * weights[i][j]
            if not (-2**15 <= product <= 2**15 - 1):
                raise OverflowError("Product outside INT16 range")

            # MAC operation
            ACC += product

            # Check ACC after EVERY MAC
            if not (-2**31 <= ACC <= 2**31 - 1):
                raise OverflowError("ACC overflow")

        print("MAC", i, "ACC =", ACC)

        # Add bias
        z = ACC + bias[i]

        # Check result
        if not (-2**31 <= z <= 2**31 - 1):
            raise OverflowError("z overflow")

        # ReLU
        if z > 0:
            y = z
        else:
            y = 0

        outputs.append(y)
        # print("Neuron", i)
        # print("ACC =", ACC)
        # print("z =", z)
        # print("ReLU =", y)
        # print()

    return outputs
