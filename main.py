import argparse
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from flox import federated_fit
from flox.endpoint import EndpointKind
from flox.logger import *
from pathlib import Path
from tensorflow import keras
from typing import Optional


def _init_endpoints(path: Path) -> tuple[list, EndpointKind]:
    endpoints = yaml.safe_load(open(path, "r"))
    endp_data_types = list()
    for endp, endp_val in endpoints.items():
        for key in ("funcx-id", "proxystore-id"):
            if key not in endp_val:
                raise ValueError(f"Endpoint {endp} in {path} is missing required key, {key}.")
            endp_data_types.append(endp_val[key] is None)
    
    if all([data_type is None for data_type in endp_data_types]):
        kind = EndpointKind.local
    elif all([data_type is not None for data_type in endp_data_types]):
        kind = EndpointKind.remote
    else:
        raise ValueError("Mixed key data types. Keys for a .yaml file must be ALL be null; "
                         "otherwise they all must be not NULL.")
    
    return endpoints, kind

def create_model(
        input_shape: tuple,
        n_hidden_layers: int,
        n_classes: int,
        optimizer,
        loss,
        metrics
) -> tf.keras.Model:
    layers = [
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
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


def get_store_iters(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    store_args = {"no_store": {
        "use_proxystore": False,
        "store_name": None,
        "proxystore_dir": None,
        "metrics": True,
    }}
    if not args.no_proxystore:
        store_args["proxystore"] = {
            "use_proxystore": True,
            "store_name": args.store_name,
            "proxystore_dir": args.store_dir,
            "metrics": True
        }
    return store_args


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", "-E", default=Path("configs/pi-endpoints.yml"), type=str)
    parser.add_argument("--samples", "-s", default=100, type=int)
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--loops", "-l", default=5, type=int)
    parser.add_argument("--time_interval", "-t", default=5, type=int)
    parser.add_argument("--store-name", "-n", default="floxystore", type=str)
    parser.add_argument("--store-dir", "-d", default=None, type=str)
    parser.add_argument("--store-port", "-p", default=6060, type=int)
    parser.add_argument("--no-proxystore", action="store_true")
    parser.add_argument("--data-name", type=str, choices=["fashion_mnist", "mnist"], default="mnist")
    args = parser.parse_args()
    # if args.no_proxystore:
    #     args.store_name, args.store_dir, args.store_port = None, None, None
    # store_args = (args.store_name, args.store_dir, args.store_port)
    # if not any([
    #     all(arg is None for arg in store_args),
    #     all(arg is not None for arg in store_args)
    # ]):
    #     raise ValueError("Arguments for ProxyStore (`--store_name`, `--store_hostname`, "
    #                      "`--store_port`) must ALL be either None or not None.")
    return args


def get_test_data(data_name: str, data_dir: Optional[Path] = None) -> tuple[tuple, tuple]:
    legal_datasets = ["mnist", "fashion_mnist"]
    if data_name not in legal_datasets:
        raise Exception(f"Please select one of the built-in Keras datasets: {legal_datasets}.")
    if data_dir is None:
        data_dir = Path.home() / "data" / data_name
    x_test = np.load(data_dir / "x_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    shape_map = {
        "mnist": (28, 28, 1),
        "fashion_mnist": (28, 28, 1),
    }
    return (x_test, y_test), shape_map[data_name]


def main(args: argparse.Namespace) -> None:
    # Load the endpoints and set up the ProxyStore arguments for the Store.
    endpoints, endpoint_kind = _init_endpoints(args.endpoints)
    (x_test, y_test), input_shape = get_test_data(args.data_name)

    result_list = []
    for store_key, store_args in get_store_iters(args).items():
        for nhl in [1]: #, 5, 10, 15, 20, 25]:
            model = create_model(
                input_shape,
                n_hidden_layers=nhl,
                n_classes=10,
                optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=["accuracy"]
            )

            # Start the FL loop.
            results = federated_fit(
                global_model=model,  # .to_json(),
                endpoints=endpoints,
                endpoint_kind=endpoint_kind,
                num_samples=args.samples,
                epochs=args.epochs,
                loops=args.loops,
                time_interval=args.time_interval,
                keras_dataset=args.data_name,
                preprocess=False,
                input_shape=input_shape,  # (32, 32, 32, 3),
                optimizer="adam",
                metrics=["accuracy"],
                x_test=x_test,
                y_test=y_test,
                use_proxystore=store_args["use_proxystore"], # Store arguments
                store_name=store_args["store_name"],
                proxystore_dir=store_args["proxystore_dir"],
            )
            results["num_hidden_layers"] = nhl
            results["store"] = store_key
            result_list.append(results)
            logging.info(f"(# hidden layers = {nhl})  Received training results: {results}")

    data = pd.concat(result_list)
    timestamp = datetime.datetime.now().isoformat().replace("T", "_")
    data.to_csv(Path(f"out/results_{timestamp}.csv"), index=False)


if __name__ == '__main__':
    main(get_args())
