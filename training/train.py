import torch
import torch.nn as nn
from tiny_network import tinyNetwork

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


with torch.no_grad():
    outputs = model(inputs)

predictions = torch.argmax(outputs, dim=1)
print("Predictions:", predictions)
print("Labels:", labels)

print(model.layer1.weight)
print(model.layer1.bias)

print(model.layer2.weight)
print(model.layer2.bias)