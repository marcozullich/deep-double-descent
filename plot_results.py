import torch
from matplotlib import pyplot as plt
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help=".pt files with the results dict")
    args = parser.parse_args()
    # result_dict_1 = torch.load("results_cnn_kmnist.pt")
    # result_dict_2 = torch.load("results_cnn_kmnist2.pt")
    # result_dict = {**result_dict_1, **result_dict_2}
    # result_dict = result_dict_1
    results_dict = torch.load(args.file)
    widths = sorted([k for k in result_dict.keys()])
    train = []
    test = []

    for width in widths:
        train.append(result_dict[width]["train"])
        test.append(result_dict[width]["test"])

    plt.plot(widths, np.log(train))
    plt.plot(widths, np.log(test))
    #plt.show()
    plt.savefig("results.png")
