def run_golden_model(sample, weights1, bias1, weights2, bias2, scales):

    # ------------------------------------------------------------
    # Quantize input sample: FP32/input values -> INT8
    # ------------------------------------------------------------

    q_sample = []

    for x in sample:
        q = round(x / scales["input_scale"])

        if q > 127:
            q = 127
        elif q < -127:
            q = -127

        q_sample.append(q)


    # ------------------------------------------------------------
    # Layer 1
    # INT8 input x INT8 weight -> INT32 accumulation
    # Add INT32 bias
    # Apply ReLU
    # ------------------------------------------------------------

    layer1_output = []

    for i in range(len(weights1)):
        ACC = 0

        for j in range(len(q_sample)):
            product = q_sample[j] * weights1[i][j]
            ACC += product

        z = ACC + bias1[i]

        # ReLU
        if z > 0:
            y = z
        else:
            y = 0

        layer1_output.append(y)


    # ------------------------------------------------------------
    # Requantize Layer 1
    # INT32 output -> INT8 input for Layer 2
    # ------------------------------------------------------------

    layer1_current_scale = (
        scales["input_scale"]
        * scales["weight_scale1"]
    )

    layer1_next_scale = scales["layer1_activation_scale"]

    requant_multiplier = (
        layer1_current_scale
        / layer1_next_scale
    )

    layer1_int8 = []

    for value in layer1_output:
        q = round(value * requant_multiplier)

        if q > 127:
            q = 127
        elif q < -127:
            q = -127

        layer1_int8.append(q)


    # ------------------------------------------------------------
    # Layer 2
    # INT8 input x INT8 weight -> INT32 accumulation
    # Add INT32 bias
    # No ReLU because these are the final logits
    # ------------------------------------------------------------

    layer2_output = []

    for i in range(len(weights2)):
        ACC = 0

        for j in range(len(layer1_int8)):
            product = layer1_int8[j] * weights2[i][j]
            ACC += product

        z = ACC + bias2[i]

        layer2_output.append(z)


    # ------------------------------------------------------------
    # Dequantize final Layer 2 output
    # INT32 logits -> approximate FP32 logits
    # ------------------------------------------------------------

    layer2_scale = (
        scales["layer1_activation_scale"]
        * scales["weight_scale2"]
    )

    layer2_float = []

    for value in layer2_output:
        layer2_float.append(
            value * layer2_scale
        )


    # ------------------------------------------------------------
    # Classification
    # Highest logit determines predicted class
    # ------------------------------------------------------------

    prediction = layer2_float.index(
        max(layer2_float)
    )


    # ------------------------------------------------------------
    # Return intermediate values for verification / RTL testing
    # ------------------------------------------------------------

    return {
        "quantized_input": q_sample,
        "layer1_int32": layer1_output,
        "layer1_int8": layer1_int8,
        "layer2_int32": layer2_output,
        "logits": layer2_float,
        "prediction": prediction
    }