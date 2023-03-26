import argparse
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml

from flox import federated_fit
from flox.logger import *
from pathlib import Path
from tensorflow import keras

plt.style.use("bmh")


def create_model(
        n_hidden_layers: int,
        n_classes: int,
        optimizer,
        loss,
        metrics
) -> tf.keras.Model:
    layers = [
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2))
    ]
    for _ in range(n_hidden_layers):
        layers.extend([
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
        ])
    layers.extend([
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_classes, activation="softmax")
    ])

    model = keras.Sequential(layers)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", "-E", default=Path("configs/local-endpoints.yml"), type=str)
    parser.add_argument("--samples", "-s", default=100, type=int)
    parser.add_argument("--epochs", "-e", default=3, type=int)
    parser.add_argument("--loops", "-l", default=5, type=int)
    parser.add_argument("--time_interval", "-t", default=5, type=int)
    parser.add_argument("--store-name", "-n", default="floxystore", type=str)
    parser.add_argument("--store-hostname", "-o", default="localhost", type=str)
    parser.add_argument("--store-port", "-p", default=6060, type=int)
    parser.add_argument("--no-proxystore", action="store_true")
    args = parser.parse_args()
    if args.no_proxystore:
        args.store_name, args.store_hostname, args.store_port = None, None, None
    store_args = (args.store_name, args.store_hostname, args.store_port)
    if not any([
        all(arg is None for arg in store_args),
        all(arg is not None for arg in store_args)
    ]):
        raise ValueError("Arguments for ProxyStore (`--store_name`, `--store_hostname`, "
                         "`--store_port`) must ALL be either None or not None.")
    return args


def main(args: argparse.Namespace) -> None:
    # Load the endpoints and set up the ProxyStore arguments for the Store.
    endpoint_ids = yaml.safe_load(open(args.endpoints, 'r'))
    store_args = dict(name=args.store_name, hostname=args.store_hostname, port=args.store_port)
    if None in store_args.values():
        store_args = dict(
            name=None,
            hostname=None,
            port=None
        )

    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()

    result_list = []
    for nhl in [1, 5, 10]:
        model = create_model(
            n_hidden_layers=nhl,
            n_classes=10,
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        # Start the FL loop.
        results = federated_fit(
            global_model=model,  # .to_json(),
            endpoint_ids=endpoint_ids,
            num_samples=args.samples,
            epochs=args.epochs,
            loops=args.loops,
            time_interval=args.time_interval,
            keras_dataset="cifar10",
            preprocess=False,
            path_dir="/home/pi/datasets",
            x_train_name="mnist_x_train.npy",
            y_train_name="mnist_y_train.npy",
            input_shape=(32, 32, 32, 3),
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
            x_test=x_test,
            y_test=y_test,
            store_args=store_args
        )
        results["num_hidden_layers"] = nhl
        result_list.append(results)
        logging.info(f"(# hidden layers = {nhl})  Received training results: {results}")

    data = pd.concat(result_list)
    timestamp = datetime.datetime.now().isoformat().replace("T", "_")
    data.to_csv(Path(f"out/results_{timestamp}.csv"))
    sns.lineplot(data=data, x="round", y="accuracy", hue="endpoint_id", style="num_hidden_layers")
    plt.show()

    sns.lineplot(data=data, x="round", y="transfer_time", hue="endpoint_id", style="num_hidden_layers")
    plt.show()


if __name__ == '__main__':
    main(get_args())
