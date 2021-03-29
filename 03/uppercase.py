#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

# a507688b-17c7-11e8-9de3-00505601122b
# bee39584-17d2-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, notably
# for `alphabet_size` and `window` and others.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=50, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=6, type=int, help="Window size to use.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout regularization.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--hidden_layers", default=[200, 200], nargs="*", type=int, help="Hidden layer sizes.")

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

    train = False      # True - train a model
    testing = False     # True - cropping train data for debugging the whole thing

    if testing:
        uppercase_data.train.data["windows"] = uppercase_data.train.data["windows"][0:1000]
        uppercase_data.train.data["labels"] = uppercase_data.train.data["labels"][0:1000]

    if args.label_smoothing == 0:
        loss = tf.losses.SparseCategoricalCrossentropy()
        metrics = tf.metrics.SparseCategoricalAccuracy(name="accuracy")
    else:
        loss = tf.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        metrics = tf.metrics.CategoricalAccuracy(name="accuracy")
        uppercase_data.train.data["labels"] = tf.keras.utils.to_categorical(uppercase_data.train.data["labels"])
        uppercase_data.dev.data["labels"] = tf.keras.utils.to_categorical(uppercase_data.dev.data["labels"])
        uppercase_data.test.data["labels"] = tf.keras.utils.to_categorical(uppercase_data.test.data["labels"])


    def create_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32))
        model.add(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))))
        model.add(tf.keras.layers.Flatten())
        if args.dropout > 0:
            model.add(tf.keras.layers.Dropout(args.dropout))
        for hidden_layer in args.hidden_layers:
            model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu))
            if args.dropout > 0:
                model.add(tf.keras.layers.Dropout(args.dropout))
        model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

        return model

    def compile(model):
        model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=loss,
            metrics=[metrics],
        )

    if train:
        model = create_model()
        model.summary()
        compile(model)
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
