import torch

from pathlib import Path
from tiny_network import tinyNetwork
from quantize import (quantizeWeights, quantizeActivations, quantizeBiases)


# ------------------------------------------------------------
# Recreate model and load trained parameters
# ------------------------------------------------------------

model = tinyNetwork()

state_dict = torch.load("../artifacts/trained_model.pt")

model.load_state_dict(state_dict)
model.eval()


# ------------------------------------------------------------
# Calibration data
# Used to determine activation scales
# ------------------------------------------------------------

calibration_inputs = torch.tensor([
    [1, 0, 2, 1, 0, 1, 2, 0],
    [6, 7, 5, 8, 6, 7, 5, 8],
    [0, 1, 0, 2, 1, 0, 1, 0],
    [8, 6, 7, 5, 8, 7, 6, 7]
], dtype=torch.float32)


# ------------------------------------------------------------
# Quantize network input
# ------------------------------------------------------------

q_inputs, input_scale = quantizeActivations(
    calibration_inputs
)


# ------------------------------------------------------------
# Layer 1
# ------------------------------------------------------------

# Quantize Layer 1 weights
q_weights1, weight_scale1 = quantizeWeights(
    model.layer1.weight.detach()
)

# Run FP32 Layer 1 + ReLU to determine the activation range
with torch.no_grad():
    layer1_fp = model.relu(
        model.layer1(calibration_inputs)
    )

# Quantize Layer 1 output activations
q_layer1_activations, layer1_activation_scale = quantizeActivations(
    layer1_fp
)

# Quantize Layer 1 bias
# Bias scale = input scale * weight scale
q_bias1, bias_scale1 = quantizeBiases(
    model.layer1.bias.detach(),
    input_scale,
    weight_scale1
)


# ------------------------------------------------------------
# Layer 2
# ------------------------------------------------------------

# Quantize Layer 2 weights
q_weights2, weight_scale2 = quantizeWeights(
    model.layer2.weight.detach()
)

# Quantize Layer 2 bias
# Layer 2 input is the output activation from Layer 1
q_bias2, bias_scale2 = quantizeBiases(
    model.layer2.bias.detach(),
    layer1_activation_scale,
    weight_scale2
)


# ------------------------------------------------------------
# Print scales
# ------------------------------------------------------------

print("----- Scales -----")

print("Input scale:")
print(input_scale)

print("Layer 1 weight scale:")
print(weight_scale1)

print("Layer 1 bias scale:")
print(bias_scale1)

print("Layer 1 activation scale:")
print(layer1_activation_scale)

print("Layer 2 weight scale:")
print(weight_scale2)

print("Layer 2 bias scale:")
print(bias_scale2)


# ------------------------------------------------------------
# Print quantized parameters
# ------------------------------------------------------------

print("\n----- Layer 1 -----")

print("INT8 weights:")
print(q_weights1)
print(q_weights1.dtype)

print("INT32 bias:")
print(q_bias1)
print(q_bias1.dtype)


print("\n----- Layer 2 -----")

print("INT8 weights:")
print(q_weights2)
print(q_weights2.dtype)

print("INT32 bias:")
print(q_bias2)
print(q_bias2.dtype)


# ------------------------------------------------------------
# Optional calibration inspection
# ------------------------------------------------------------

print("\n----- Calibration -----")

print("Quantized network inputs:")
print(q_inputs)

print("Quantized Layer 1 activations:")
print(q_layer1_activations)

# ------------------------------------------------------------
# Export quantized model parameters
# ------------------------------------------------------------

with open("../artifacts/layer1_weights.txt", "w") as f:
    for row in q_weights1:
        f.write(" ".join(str(int(x)) for x in row) + "\n")

with open("../artifacts/layer1_bias.txt", "w") as f:
    f.write(" ".join(str(int(x)) for x in q_bias1))

with open("../artifacts/layer2_weights.txt", "w") as f:
    for row in q_weights2:
        f.write(" ".join(str(int(x)) for x in row) + "\n")

with open("../artifacts/layer2_bias.txt", "w") as f:
    f.write(" ".join(str(int(x)) for x in q_bias2))

with open("../artifacts/scales.txt", "w") as f:
    f.write(f"input_scale {float(input_scale)}\n")
    f.write(f"weight_scale1 {float(weight_scale1)}\n")
    f.write(f"bias_scale1 {float(bias_scale1)}\n")
    f.write(f"layer1_activation_scale {float(layer1_activation_scale)}\n")
    f.write(f"weight_scale2 {float(weight_scale2)}\n")
    f.write(f"bias_scale2 {float(bias_scale2)}\n")

print("\nQuantized model exported to artifacts/")