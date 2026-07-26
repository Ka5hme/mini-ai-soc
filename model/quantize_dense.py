import torch

def quantized_dense(sample, weights, bias, apply_relu=True):
    outputs = []
    for i in range(len(weights)):
        ACC = torch.tensor(0, dtype=torch.int32)
        for j in range(len(sample)):
            product = sample[j].to(torch.int32) * weights[i][j].to(torch.int32)
            # MAC operation
            ACC += product

        # Add bias
        z = ACC + bias[i]

        # ReLU
        if apply_relu:
            if z > 0:
                y = z
            else:
                y = torch.tensor(0, dtype=torch.int32)
        else:
            y = z

        outputs.append(y)


    return outputs