#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, notably
# for `alphabet_size` and `window` and others.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=50, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=6, type=int, help="Window size to use.")

def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    train = False       # True - train a model
    testing = False     # True - cropping train data for debugging the whole thing

    if testing:
        uppercase_data.train.data["windows"] = uppercase_data.train.data["windows"][0:1000]
        uppercase_data.train.data["labels"] = uppercase_data.train.data["labels"][0:1000]

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters and train the model.
    #
    # The inputs are _windows_ of fixed size (`args.window` characters on left,
    # the character in question, and `args.window` characters on right), where
    # each character is represented by a `tf.int32` index. To suitably represent
    # the characters, you can:
    # - Convert the character indices into _one-hot encoding_. There is no
    #   explicit Keras layer, but you can
    #   - use a Lambda layer which can encompass any function:
    #       tf.keras.Sequential([
    #         tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
    #         tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
    #   - or use Functional API and then any TF function can be used
    #     as a Keras layer:
    #       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
    #       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
    #   You can then flatten the one-hot encoded windows and follow with a dense layer.
    # - Alternatively, you can use `tf.keras.layers.Embedding` (which is an efficient
    #   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.
    def create_model():
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
            tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=2, activation=tf.nn.softmax),
        ])
        return model
    def compile(model):
        model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
        )

    if train:
        model = create_model()
        model.summary()
        model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
        )

        model.fit(
            uppercase_data.train.data["windows"], uppercase_data.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"]),
        )

        model.evaluate(
            uppercase_data.test.data["windows"], uppercase_data.test.data["labels"], batch_size=args.batch_size,
        )
        model.save_weights('models/uppercase_model')
    else:
        model = create_model()
        compile(model)
        model.load_weights('models/uppercase_model')
        model.evaluate(
            uppercase_data.test.data["windows"], uppercase_data.test.data["labels"], batch_size=args.batch_size,
        )

    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to predictions_file (which is
    # `uppercase_test.txt` in the `args.logdir` directory).

    predicted = model.predict_classes(uppercase_data.test.data["windows"])

    with open( "uppercase_test.txt", "w", encoding="utf-8") as predictions_file:
    # with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        for i in range(uppercase_data.test.size):
            if predicted[i] == 0:
               # print("%s" % (char_from_text), end="")
                print("%s" % (uppercase_data.test.text[i]), end="", file=predictions_file)
            else:
               # print("%s" % (char_from_text.upper()), end="")
                print("%s" % (uppercase_data.test.text[i].upper()), end="", file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
