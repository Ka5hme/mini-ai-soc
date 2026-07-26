def load_matrix(filename):
    matrix = []

    with open(filename, "r") as f:
        for line in f:
            row = [int(x) for x in line.split()]
            matrix.append(row)

    return matrix


def load_vector(filename):
    with open(filename, "r") as f:
        vector = [int(x) for x in f.read().split()]

    return vector


def load_scales(filename):
    scales = {}

    with open(filename, "r") as f:
        for line in f:
            name, value = line.split()
            scales[name] = float(value)

    return scales

