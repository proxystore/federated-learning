import dataclasses
import globus_sdk
import numpy as np
import pickle
import proxystore.connectors.endpoint
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
from proxystore.proxy import Proxy
from proxystore.store import Store, register_store
from tensorflow import keras
from typing import Optional

DEFAULT_STORE_NAME = "floxystore"
FUNCX_SIZE_LIMIT = 10485760


def _init_store(
        use_proxystore: bool = False,
        store_name: str = DEFAULT_STORE_NAME,
        proxystore_endpoints: Optional[list[str]] = None,
        proxystore_dir: Path = None,
        metrics: bool = True
) -> Optional[Store]:
    try:
        logging.info(f"Attempting to initialize Store with the following endpoints: {proxystore_endpoints}.")
        if use_proxystore:
            store = Store(
                name=store_name,
                connector=EndpointConnector(proxystore_endpoints, proxystore_dir),
                metrics=metrics
            )
            register_store(store)
            return store
    except proxystore.connectors.endpoint.EndpointConnectorError as err:
        logging.error(f"The provided endpoint IDs are:\n{proxystore_endpoints}")
        raise err
    return None


def federated_fit(
        fxe: FuncXExecutor,
        train_fn_uuid,
        global_model,
        endpoints: dict[str, dict[str, str]],
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
        use_proxystore: bool = True,  # Store arguments
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
        False
    )
    results = defaultdict(list)

    # Global Federated training loop.
    logging.info("Initializing `FuncXExecutor` and registering local training function.")
    for rnd in range(loops):
        # Save the model architecture, weights, and prepare to store tasks/results for execution.
        model_json = global_model.to_json()
        global_model_weights = np.asarray(global_model.get_weights(), dtype=object)
        futures, round_results, endp_end_times = list(), list(), dict()
        logging.info(f"Starting round {rnd + 1}/{loops}.")

        for endp, samples, n_epochs, path in zip(*iters):
            if endp == "controller":
                continue

            weights = store.proxy(global_model_weights, evict=False) if use_proxystore else global_model_weights
            model_json = store.proxy(model_json, evict=False) if use_proxystore else model_json
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
                round_results.append(local_fit(**kwargs))
                endp_end_times[endp] = time.time()
            else:
                if sys.getsizeof(pickle.dumps(kwargs)) < FUNCX_SIZE_LIMIT:
                    endp_uuid = endpoints[endp]["funcx-id"]
                    logging.info(f"Submitting local training job via FuncX to endpoint '{endp_uuid}'.")
                    try:
                        fxe.endpoint_id = endp_uuid
                        fut = fxe.submit_to_registered_function(train_fn_uuid, kwargs=kwargs)
                        futures.append(fut)
                    except FuncxError:
                        pass

        # Retrieve and store the results from the training nodes.
        logging.info("Retrieving results from the endpoint(s).")
        if endpoint_kind is EndpointKind.remote:
            for fut in futures:
                try:
                    res = fut.result()
                    if res is None:
                        continue
                    res["end_transfer_time"] = time.time()
                    round_results.append(res)
                except globus_sdk.exc.api.GlobusAPIError:
                    pass
                except MaxResultSizeExceeded:
                    pass

        if len(round_results) == 0:
            if time_interval > 0:
                time.sleep(time_interval)
            continue

        # Extract model updates from endpoints and then aggregate them to update the global model.
        logging.info(f"Starting the model aggregation phase {rnd + 1}.")
        model_weights = [res["model_weights"] for res in round_results]
        weights = np.array([res["samples_count"] for res in round_results])
        weights = weights / weights.sum(0)
        average_weights = np.average(model_weights, weights=weights, axis=0)
        global_model.set_weights(average_weights)
        logging.info(f"Finished model aggregation phase {rnd + 1}.")

        # Evaluate the global model, if all the necessary parameters are given.
        if all([x_test is not None, y_test is not None]):
            logging.info("Beginning the global testing phase.")
            test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)
            logging.info(f"Finished the global testing phase:  {test_loss=:0.4f}  |  {test_acc=:0.5f}.")
        else:
            test_loss, test_acc = None, None

        # Record the results from this aggregation round.
        logging.info(f"Storing the results from aggregation round {rnd + 1}.")
        for res in round_results:
            if history_metrics_aggr == "mean":
                train_acc = sum(res["accuracy"]) / len(res["accuracy"])
                test_loss = sum(res["loss"]) / len(res["loss"])
            else:  # i.e., history_metrics_aggr == "recent"
                train_acc = res["accuracy"][-1]
                test_loss = res["loss"][-1]

            results["round"].append(rnd + 1)
            results["endpoint_name"].append(res["endpoint_name"])
            results["accuracy"].append(train_acc)
            results["loss"].append(test_loss)
            results["transfer_time"].append(res["end_transfer_time"] - res["start_transfer_time"])
            results["model_size_bytes"].append(sys.getsizeof(global_model_weights))
            results["local"].append(endpoint_kind is EndpointKind.local)
            results["test_accuracy"].append(test_acc)
            results["test_loss"].append(test_loss)

        # If `time_interval` is supplied, wait for `time_interval` seconds.
        if time_interval > 0:
            time.sleep(time_interval)

    return DataFrame.from_dict(results)
