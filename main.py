import numpy as np
import math as m
import nn


def gen_data_bin_class(f, size=100):
    data = []
    for i in range(size):
        x1 = np.random.uniform(0, 1)
        x2 = np.random.uniform(0, 1)
        data_point = [x1, x2, int(f(x1) <= x2)]
        data.append(data_point)
    return data


def main():
    q2_data = gen_data_bin_class(lambda x: x**2 * m.sin(2 * m.pi * x) + 0.7)

    network = nn.init_network([2, 1])
    for dp in q2_data:
        x = np.array(dp[:2])
        actual_class = dp[2]
        print(nn.eval_network(network, x, lambda x: 1 / (1 + m.exp(-x))))
        # dp_grads = nn.bprop_network(network, x, actual_class, lambda x: 1 / (1 + m.exp(-x)))
        # print(dp_grads)


if __name__ == "__main__":
    main()
