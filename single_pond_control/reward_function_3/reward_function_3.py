def reward_function_3(depth, outflow, gate_postions_rate):
    """reward_function_3

    :param depth:
    :param outflow:
    :param gate_postions_rate:
    """
    # outflow
    depth = float(depth)
    outflow = float(outflow)
    gate_postions_rate = float(gate_postions_rate)
    reward_flow = 1.0 if outflow < 0.1 else -1.0
    reward_depth = -0.5*depth if depth < 2.0 else -depth**2 + 3.0
    reward_gate = 0.0 if gate_postions_rate < 0.4 else -10.0*gate_postions_rate

    return reward_flow + reward_depth + reward_gate
