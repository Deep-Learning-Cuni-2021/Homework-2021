#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    dict_data = {}
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            dict_data[line] = dict_data.get(line, 0) + 1

    arr_data = np.empty(len(dict_data))
    sum_data = 0
    x = 0

    for key in dict_data:
        arr_data[x] = dict_data[key]
        sum_data += dict_data[key]
        x += 1

    arr_data = arr_data / sum_data

    # TODO: Load model distribution, each line `string \t probability`.
    dict_model = {}
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            key, value = line.split()  # Split line into a tuple
            dict_model[key] = float(value)  # Add tuple values to dictionary
    # TODO: process the line, aggregating using Python data structures

    # TODO: Create a NumPy array containing the model distribution.
    arr_model = np.empty(len(dict_data))
    x = 0
    for key in dict_data:
        arr_model[x] = dict_model.get(key, np.inf)    # using `np.inf` when needed
        x += 1

    # TODO: Compute the entropy H(data distribution).
    entropy = -np.sum(arr_data * np.log(arr_data))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf` (this is done when making model array)
    crossentropy = -np.sum(arr_data * np.log(arr_model))

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution)
    kl_divergence = np.sum(arr_data * np.log(arr_data / arr_model))

    # Return the computed values for ReCodEx to validate

    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
