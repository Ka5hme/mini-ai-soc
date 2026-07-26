from loaders import load_matrix, load_vector, load_scales
from golden_model import run_golden_model


weights1 = load_matrix("../artifacts/layer1_weights.txt")
bias1 = load_vector("../artifacts/layer1_bias.txt")

weights2 = load_matrix("../artifacts/layer2_weights.txt")
bias2 = load_vector("../artifacts/layer2_bias.txt")

scales = load_scales("../artifacts/scales.txt")


sample = [6, 7, 5, 8, 6, 7, 5, 8]


results = run_golden_model(
    sample,
    weights1,
    bias1,
    weights2,
    bias2,
    scales
)


print("----- Golden Model Results -----")

print("Input:", sample)
print("Quantized input:", results["quantized_input"])

print("Layer 1 INT32:", results["layer1_int32"])
print("Layer 1 INT8:", results["layer1_int8"])

print("Layer 2 INT32:", results["layer2_int32"])

print("Final logits:", results["logits"])
print("Prediction:", results["prediction"])