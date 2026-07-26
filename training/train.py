import torch
import torch.nn as nn
from tiny_network import tinyNetwork
from quantize import quantizeWeights
from quantize import quantizeActivations
from quantize import quantizeBiases
from quantize import requantizeActivations
from model.quantize_dense import quantized_dense

# 4 sample values 8 input represented as a PyTorch tensor
inputs = torch.tensor([
    [1, 0, 2, 1, 0, 1, 2, 0],
    [6, 7, 5, 8, 6, 7, 5, 8],
    [0, 1, 0, 2, 1, 0, 1, 0],
    [8, 6, 7, 5, 8, 7, 6, 7]
], dtype=torch.float32)
labels = torch.tensor([0, 1, 0, 1])

model = tinyNetwork()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    # Clear gradients from previous iteration
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)

    # Calculate error
    loss = loss_function(outputs, labels)

    # Calculate gradients
    loss.backward()

    # Update weights and biases
    optimizer.step()

    # Print progress occasionally
    if epoch % 10 == 0:
        print("Epoch:", epoch, "Loss:", loss.item())

    # print("Input shape:", inputs.shape)
    # print("Output shape:", outputs.shape)
    # print("Outputs:", outputs)
    # print("Loss:", loss)
    # print(model.layer1.weight.grad)
    # print('--------------------------------------------------------')

q_inputs, input_scale = quantizeActivations(inputs)

print("Quantized inputs:")
print(q_inputs)

print("Input scale:", input_scale)
print("Input dtype:", q_inputs.dtype)

q_weights, weight_scale = quantizeWeights(model.layer1.weight.detach())

print("Float weights:")
print(model.layer1.weight.detach())

print("Quantized weights:")
print(q_weights)
print(q_weights.dtype)

print("Scale:")
print(weight_scale)

q_bias, bias_scale = quantizeBiases(model.layer1.bias.detach(), input_scale, weight_scale)

print("Bias:")
print(q_bias)
print(q_bias.dtype)
print(bias_scale)

q_layer1_output = quantized_dense(q_inputs[1], q_weights, q_bias, apply_relu=True)


q_layer1_output = torch.stack(q_layer1_output)

print("Quantized Layer 1 output:")
print(q_layer1_output)

layer1_scale = input_scale * weight_scale
dequantized = q_layer1_output.to(torch.float32) * layer1_scale

q_layer1_activated, layer1_activation_scale = requantizeActivations(q_layer1_output, layer1_scale)


q_weights2, weight_scale2 = quantizeWeights(model.layer2.weight.detach())
q_bias2, bias_scale2 = quantizeBiases(model.layer2.bias.detach(), layer1_activation_scale, weight_scale2)
q_layer2_output = quantized_dense(q_layer1_activated, q_weights2, q_bias2, apply_relu=False)
q_layer2_output = torch.stack(q_layer2_output)
layer2_scale = layer1_activation_scale * weight_scale2
dequantized_layer2 = (
    q_layer2_output.to(torch.float32)
    * layer2_scale
)

print("Layer 1 INT32:")
print(q_layer1_output)

print("Layer 1 requantized INT8:")
print(q_layer1_activated)

print("Layer 1 activation scale:")
print(layer1_activation_scale)

print("Layer 1 activation dtype:")
print(q_layer1_activated.dtype)

print(model.relu(model.layer1(inputs)))

with torch.no_grad():
    fp_output = model(inputs[1])

print("Layer 2 INT32:")
print(q_layer2_output)

print("Layer 2 dequantized:")
print(dequantized_layer2)

print("FP32 model output:")
print(fp_output)


predictions = torch.argmax(outputs, dim=1)
