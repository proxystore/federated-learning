import numpy as np
import time

from collections import defaultdict
from flox.endpoint_logic import local_fit
from flox.logger import *
from flox.results import FloxResults
from funcx.sdk.executor import FuncXExecutor
from pandas import DataFrame
from tensorflow import keras
from time import perf_counter
from typing import Optional


def federated_fit(
        global_model,
        endpoint_ids,
        num_samples=100,
        epochs=5,
        loops=1,
        time_interval=0,
        keras_dataset: str = "mnist",
        preprocess: bool = False,
        path_dir: Optional[str] = None,
        x_train_name="mnist_x_train.npy",
        y_train_name="mnist_y_train.npy",
        input_shape=(32, 28, 28, 1),
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=None,
        x_test=None,
        y_test=None,
        store_args: dict[str, str] = None,  # ProxyStore Store object
        history_metrics_aggr: str = None,
        silent: bool = False
) -> DataFrame:
    # Handle default argument values.
    if isinstance(num_samples, int):
        num_samples = [num_samples] * len(endpoint_ids)
    if isinstance(epochs, int):
        epochs = [epochs] * len(endpoint_ids)
    if path_dir is None:
        path_dir = "/home/pi/datasets"
    if isinstance(path_dir, str):
        path_dir = [path_dir] * len(endpoint_ids)
    if history_metrics_aggr is None:
        history_metrics_aggr = "mean"
    if metrics is None:
        metrics = ["accuracy"]
    if history_metrics_aggr not in ["mean", "recent"]:
        raise ValueError("Argument `history_metrics_aggr` must be either 'mean' or 'recent'.")

    iters = (endpoint_ids, num_samples, epochs, path_dir)
    if not all([len(endpoint_ids) == len(it) for it in iters]):
        raise ValueError(f"Length of iterators ('endpoint_ids', 'num_samples', "
                         f"'epochs', 'path_dir') must be of the same length.")

    # Initialize the data indices for each of the training endpoints.
    (x_train, _), (_, _) = keras.datasets.cifar10.load_data()
    train_indices = {
        endp: np.random.choice(np.arange(len(x_train)), k, replace=True)
        for endp, k in zip(endpoint_ids, num_samples)
    }

    # Global Federated training loop.
    results = defaultdict(list)
    for rnd in range(loops):
        # Save the model architecture, weights, and prepare to store tasks/results for execution.
        model_json = global_model.to_json()
        global_model_weights = np.asarray(global_model.get_weights(), dtype=object)
        tasks, local_results = [], []

        # Identify the execution kind.
        local_endpoints = list(filter(lambda val: val.startswith('local'), endpoint_ids))
        if not (len(local_endpoints) == 0 or len(local_endpoints) == len(endpoint_ids)):
            raise ValueError("You cannot have endpoints of mixed types. You can only have all "
                             "local endpoints (prefixed with 'local') or all funcX endpoints.")
        locally_execute = all([endp.startswith("local") for endp in endpoint_ids])

        # submit the corresponding parameters to each endpoint for a round of FL
        logging.info(f'{iters=}')
        endpoint_end_times = {}
        for endp, samples, n_epochs, path in zip(*iters):
            kwargs = dict(
                endpoint_id=endp,
                json_model_config=model_json,
                global_model_weights=global_model_weights,
                train_indices=train_indices[endp],
                epochs=n_epochs,
                keras_dataset=keras_dataset,
                preprocess=preprocess,
                path_dir=path,
                x_train_name=x_train_name,
                y_train_name=y_train_name,
                input_shape=input_shape,
                loss=loss,
                optimizer=optimizer,
                metrics=metrics,
                store_kind="Redis",
                store_name=store_args["name"],
                store_hostname=store_args["hostname"],
                store_port=store_args["port"],
            )
            if locally_execute:
                local_results.append(local_fit(**kwargs))
                endpoint_end_times[endp] = perf_counter()
            else:
                with FuncXExecutor(endp) as fx:
                    tasks.append(fx.submit(local_fit, **kwargs))

        # Retrieve and store the results from the training nodes.
        if not locally_execute:
            for t in tasks:
                res = t.result()
                local_results.append(res)
                endpoint_end_times[res["endpoint_id"]] = perf_counter()

        # Extract model updates from endpoints and then aggregate them to update the global model.
        model_weights = [res["model_weights"] for res in local_results]
        weights = np.array([res["samples_count"] for res in local_results])
        weights = weights / weights.sum(0)
        average_weights = np.average(model_weights, weights=weights, axis=0)
        global_model.set_weights(average_weights)
        logging.info(f"Aggr. Round {rnd}, Trained Federated Model")

        # Record the results from this aggregation round.
        for res in local_results:
            if history_metrics_aggr == "mean":
                acc_result = sum(res["accuracy"]) / len(res["accuracy"])
                loss_result = sum(res["loss"]) / len(res["loss"])
            else:  # i.e., history_metrics_aggr == "recent"
                acc_result = res["accuracy"][-1]
                loss_result = res["loss"][-1]
            results["round"].append(rnd)
            results["endpoint_id"].append(res["endpoint_id"])
            results["accuracy"].append(acc_result)
            results["loss"].append(loss_result)
            results["transfer_time"].append(endpoint_end_times[res["endpoint_id"]] - res["time_before_transfer"])

        # Evaluate the global model, if all the necessary parameters are given.
        if all([x_test is not None, y_test is not None]):
            test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)
            if not silent:
                logging.info(f"Test loss: {test_loss}  |  Test accuracy: {test_acc}")

        # If `time_interval` is supplied, wait for `time_interval` seconds
        if time_interval > 0:
            time.sleep(time_interval)

    return DataFrame.from_dict(results)
