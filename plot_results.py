import torch
from matplotlib import pyplot as plt
import numpy as np
import argparse
import re
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_folder", type=str, default=".", help="Folder where the origin file(s) is (are) located")
    parser.add_argument("--origin_file", type=str, required=True, help=".pt files with the results dict as regex.")
    parser.add_argument("--image_file", type=str, default="results.png", help="image file where the plot is going to be saved")
    parser.add_argument("--mode", choices=["concat", "average"], default="concat", help="concatenate or average multiple files")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    if args.debug:
        args.verbose = True
        args.mode = "average"
        args.image_file = "results_dbg.png"
        args.origin_file = r".+kmnist_tn.+"
        args.origin_folder = "."
    # result_dict_1 = torch.load("results_cnn_kmnist.pt")
    # result_dict_2 = torch.load("results_cnn_kmnist2.pt")
    # result_dict = {**result_dict_1, **result_dict_2}
    # result_dict = result_dict_1
    # results_dict = torch.load(args.origin_file)
    matching_files = [os.path.join(args.origin_folder, fi) for fi in os.listdir(args.origin_folder) if re.match(args.origin_file, fi)]
    if args.verbose:
        print(f"matching files\n{matching_files}")
    result_dicts = [torch.load(fi) for fi in matching_files]

    if args.mode == "concat":
        result_dict = {k: v for d in result_dicts for k, v in d.items()}
    if args.mode == "average":
        widths = [w for w in result_dicts[0].keys()]
        if args.verbose:
            print(f"widths\n{widths}")
        result_dict = {w: {"train": 0.0, "test": 0.0} for w in widths}
        
        for w in widths:
            for d in result_dicts:
                result_dict[w]["train"] += d[w]["train"]
                result_dict[w]["test"] += d[w]["test"]

            result_dict[w]["train"] /= len(result_dicts)
            result_dict[w]["test"] /= len(result_dicts)
                
        if args.verbose:
            print(result_dict)
        

    widths = sorted([k for k in result_dict.keys()])
    train = []
    test = []

    for width in widths:
        train.append(result_dict[width]["train"])
        test.append(result_dict[width]["test"])

    plt.plot(widths, np.log(train))
    plt.plot(widths, np.log(test))
    #plt.show()
    plt.savefig(args.image_file)
