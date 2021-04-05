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

from cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--cnn", default='CB-32-3-1-same,CB-32-3-1-same,M-2-2,CB-64-3-1-same,CB-64-3-1-same,M-2-2,CB-128-3-1-same,CB-128-3-1-same,M-2-2,F,H-128', type=str,
                    help="CNN architecture.")
class Network(tf.keras.Model):
    def __init__(self, args):
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

        def create_layer(input, layer):
            if layer[:2] == 'C-':
                arguments = layer.split('-')
                if arguments[4][-1] == ']':
                    arguments[4] = arguments[4][:-1]
                return tf.keras.layers.Conv2D(filters=int(arguments[1]), kernel_size=int(arguments[2]),
                                                strides=(int(arguments[3]), int(arguments[3])), padding=arguments[4],
                                                activation=tf.keras.activations.relu)(input)

            elif layer[:2] == 'CB':
                arguments = layer.split('-')
                if arguments[4][-1] == ']':
                    arguments[4] = arguments[4][:-1]
                a = tf.keras.layers.Conv2D(filters=int(arguments[1]), kernel_size=int(arguments[2]),
                                                strides=(int(arguments[3]), int(arguments[3])), padding=arguments[4],
                                                activation=None, use_bias=False)(input)
                b = tf.keras.layers.BatchNormalization()(a)
                return tf.keras.activations.relu(b)

            elif layer[0] == "M":
                arguments = layer.split("-")
                return tf.keras.layers.MaxPool2D(pool_size=(int(arguments[1]), int(arguments[1])), strides=int(arguments[2]), padding='same')(input)

            elif layer[0] == "F":
                return tf.keras.layers.Flatten()(input)

            elif layer[0] == "H":
                arguments = layer.split("-")
                return tf.keras.layers.Dense(units=int(arguments[1]), activation=tf.keras.activations.relu)(input)

            elif layer[0] == "D":
                arguments = layer.split("-")
                return tf.keras.layers.Dropout(rate=float(arguments[1]))(input)

        layers_list = re.split(',', args.cnn)

        residual = False
        hidden = inputs
        for layer in layers_list:
            if layer[0] != 'R' and not residual:
                hidden = create_layer(input=hidden, layer=layer)
            elif layer[0] == 'R' and not residual:
                from_where = hidden
                hidden = create_layer(input=hidden, layer=layer[3:])
                residual = True
            elif residual:
                hidden = create_layer(input=hidden, layer=layer)
                hidden = tf.keras.layers.Add()([hidden, from_where])
                residual = False

        outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100,
                                                          profile_batch=0)
        self.tb_callback._close_writers = lambda: None  # A hack allowing to keep the writers open.


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")))

    # Load data
    cifar = CIFAR10()
    model = Network(args)
    model.summary()
    model.fit(
        cifar.train.data["images"], cifar.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
        callbacks=[model.tb_callback],
    )

    # Compute test set accuracy and return it
    test_logs = model.evaluate(
        cifar.dev.data["images"], cifar.dev.data["labels"], batch_size=args.batch_size, return_dict=True,
    )
    model.tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})



    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
