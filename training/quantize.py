import torch

def quantizeWeights(weights):
    max_abs = torch.max(torch.abs(weights))
    scale =  max_abs / 127
    q = torch.round(weights/scale)
    q = torch.clamp(q, -127, 127)
    q = q.to(torch.int8)

    return q, scale

def quantizeActivations(activations):
    max_abs = torch.max(torch.abs(activations))
    scale =  max_abs / 127
    q = torch.round(activations/scale)
    q = torch.clamp(q, -127, 127)
    q = q.to(torch.int8)

    return q, scale

def quantizeBiases(biases, input_scale, weight_scale):
    bias_scale = input_scale * weight_scale
    q = torch.round(biases / bias_scale)
    q = q.to(torch.int32)
    return q, bias_scale

def requantizeActivations(q_values, current_scale):
    float_values = q_values.to(torch.float32) * current_scale
    max_abs = torch.max(torch.abs(float_values))
    if max_abs == 0:
        q = torch.zeros_like(q_values, dtype=torch.int8)
        scale = 1.0
        return q, scale
    scale =  max_abs / 127
    q = torch.round(float_values/scale)
    q = torch.clamp(q, -127, 127)
    q = q.to(torch.int8)
    return q, scale
