import numpy as np
import sys
import time
import yaml

from collections import defaultdict
from flox.endpoint import EndpointKind
from flox.endpoint_logic import local_fit
from flox.logger import *
from flox.results import FloxResults
from funcx import FuncXClient
from funcx.errors import FuncxError, MaxResultSizeExceeded
from funcx.sdk.executor import FuncXExecutor
from pandas import DataFrame
from pathlib import Path
from proxystore.connectors.endpoint import EndpointConnector
from proxystore.store import Store
from tensorflow import keras
from time import perf_counter
from typing import Optional

DEFAULT_STORE_NAME = "floxystore"


def _init_store(
    use_proxystore: bool = False,
    store_name: str = DEFAULT_STORE_NAME,
    proxystore_endpoints: Optional[list[str]] = None,
    proxystore_dir: Path = None,
    metrics: bool = True
) -> Optional[Store]:
    if use_proxystore:
        return Store(
            name=store_name,
            connector=EndpointConnector(proxystore_endpoints, proxystore_dir),
            metrics=metrics
        )
    return None


def _confirm_local_execution(endpoints: list[str]) -> bool:
    local_endpoints = list(filter(lambda val: val.startswith("local"), endpoints))
    if not (len(local_endpoints) == 0 or len(local_endpoints) == len(endpoints)):
        raise ValueError("You cannot have endpoints of mixed types. You can only have all "
                         "local endpoints (prefixed with 'local') or all funcX endpoints.")
    return all([endp.startswith("local") for endp in endpoints])


def federated_fit(
        global_model,
        endpoints: dict[str, dict[str,str]],
        endpoint_kind: EndpointKind,
        num_samples=100,
        epochs=5,
        loops=1,
        time_interval=0,
        keras_dataset: str = "mnist",
        preprocess: bool = False,
        path_dir: Optional[str] = None,
        input_shape=(32, 28, 28, 1),
        optimizer="adam",
        metrics=None,
        x_test=None,
        y_test=None,
        history_metrics_aggr: str = None,
        use_proxystore: bool = True, # Store arguments
        store_name: str = DEFAULT_STORE_NAME,
        proxystore_dir: Path = None,
        silent: bool = True
) -> DataFrame:
    # Handle default argument values.
    if isinstance(num_samples, int):
        num_samples = [num_samples] * len(endpoints)
    if isinstance(epochs, int):
        epochs = [epochs] * len(endpoints)
    if isinstance(path_dir, str) or isinstance(path_dir, Path) or path_dir is None:
        path_dir = [path_dir] * len(endpoints)
    if history_metrics_aggr is None:
        history_metrics_aggr = "mean"
    if metrics is None:
        metrics = ["accuracy"]
    if history_metrics_aggr not in ["mean", "recent"]:
        raise ValueError("Argument `history_metrics_aggr` must be either 'mean' or 'recent'.")

    iters = (endpoints, num_samples, epochs, path_dir)
    if not all([len(endpoints) == len(it) for it in iters]):
        raise ValueError(f"Length of iterators ('endpoints', 'num_samples', "
                         f"'epochs', 'path_dir') must be of the same length.")
    
    # Initialize the data indices for each of the training endpoints.
    (x_train, _), (_, _) = keras.datasets.cifar10.load_data()
    train_indices = {
        endp: np.random.choice(np.arange(len(x_train)), k, replace=True)
        for endp, k in zip(endpoints, num_samples)
    }

    # Initialize the store to be used (if specified) and the results dictionary
    # for the entire experiment.
    proxystore_endpoints = [endp["proxystore-id"] for endp in endpoints.values()]
    store = _init_store(
        use_proxystore, 
        store_name,
        proxystore_endpoints, 
        proxystore_dir,
        True
    )
    results = defaultdict(list)

    # Global Federated training loop.
    logging.info("Initializing `FuncXExecutor` and registering local training function.")
    fx = FuncXExecutor()
    train_fn_uuid = fx.register_function(local_fit)
    for rnd in range(loops):
        # Save the model architecture, weights, and prepare to store tasks/results
        # for execution.
        model_json = global_model.to_json()
        global_model_weights = np.asarray(global_model.get_weights(), dtype=object)
        futures, local_results, endp_end_times = list(), list(), dict()

        for endp, samples, n_epochs, path in zip(*iters):
            weights = store.proxy(global_model_weights) if use_proxystore else global_model_weights
            kwargs = dict(
                endpoint_name=endp,
                json_model_config=model_json,
                global_model_weights=weights,
                train_indices=train_indices[endp],
                epochs=n_epochs,
                keras_dataset=keras_dataset,
                preprocess=preprocess,
                data_dir=path,
                input_shape=input_shape,
                optimizer=optimizer,
                metrics=metrics,
                use_proxystore=use_proxystore
            )
            if endpoint_kind is EndpointKind.local:
                local_results.append(local_fit(**kwargs))
                endp_end_times[endp] = perf_counter()
            else:
                endp_uuid = endpoints[endp]["funcx-id"]
                logging.info(f"Submitting local training job via FuncX to endpoint '{endp_uuid}'.")
                fx.endpoint_id = endp_uuid
                try:
                    fut = fx.submit_to_registered_function(train_fn_uuid, kwargs=kwargs)
                    futures.append(fut)
                except FuncxError:
                    pass


        # Retrieve and store the results from the training nodes.
        if endpoint_kind is EndpointKind.remote:
            for fut in futures:
                try:
                    res = fut.result()
                    local_results.append(res)
                    endp_end_times[res["endpoint_name"]] = perf_counter()
                except MaxResultSizeExceeded:
                    pass

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
            results["endpoint_name"].append(res["endpoint_name"])
            results["accuracy"].append(acc_result)
            results["loss"].append(loss_result)
            results["transfer_time"].append(endp_end_times[res["endpoint_name"]] - res["time_before_transfer"])
            results["model_size_bytes"].append(sys.getsizeof(global_model_weights))
            results["local"].append(endpoint_kind is EndpointKind.local)

        # Evaluate the global model, if all the necessary parameters are given.
        if all([x_test is not None, y_test is not None]):
            test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)
            if not silent:
                logging.info(f"Test loss: {test_loss}  |  Test accuracy: {test_acc}")

        # If `time_interval` is supplied, wait for `time_interval` seconds.
        if time_interval > 0:
            time.sleep(time_interval)

    fx.shutdown()
    return DataFrame.from_dict(results)
