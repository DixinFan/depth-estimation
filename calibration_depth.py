def calibrate_depth(depth):
    # (51：19) (83：22)
    slope = 11
    bias = -159
    depth = slope * depth + bias
    depth = round(depth,  -1)
    return depth