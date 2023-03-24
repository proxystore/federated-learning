import numpy as np
import time

from flox.endpoint_logic import local_fit
from flox.logger import *
from funcx.sdk.executor import FuncXExecutor
from typing import Optional

from flox.results import FloxResults


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
        silent: bool = False
) -> FloxResults:
    # Handle default argument values.
    if isinstance(num_samples, int):
        num_samples = [num_samples] * len(endpoint_ids)
    if isinstance(epochs, int):
        epochs = [epochs] * len(endpoint_ids)
    if path_dir is None:
        path_dir = "/home/pi/datasets"
    if isinstance(path_dir, str):
        path_dir = [path_dir] * len(endpoint_ids)
    if metrics is None:
        metrics = ["accuracy"]

    iters = (endpoint_ids, num_samples, epochs, path_dir)
    if not all([len(endpoint_ids) == len(it) for it in iters]):
        raise ValueError(f"Length of iterators ('endpoint_ids', 'num_samples', "
                         f"'epochs', 'path_dir') must be of the same length.")

    # Global Federated training loop.
    for i in range(loops):
        # Save the model architecture, weights, and prepare to store tasks/results for execution.
        model_json = global_model.to_json()
        global_model_weights = np.asarray(global_model.get_weights(), dtype=object)
        tasks, results = [], []

        # Identify the execution kind.
        local_endpoints = list(filter(lambda val: val.startswith('local'), endpoint_ids))
        if not (len(local_endpoints) == 0 or len(local_endpoints) == len(endpoint_ids)):
            raise ValueError("You cannot have endpoints of mixed types. You can only have all "
                             "local endpoints (prefixed with 'local') or all funcX endpoints.")
        local_run = all([endp.startswith("local") for endp in endpoint_ids])

        # submit the corresponding parameters to each endpoint for a round of FL
        logging.info(f'{iters=}')
        for endp, samples, n_epochs, path in zip(*iters):
            kwargs = dict(
                endpoint_id=endp,
                json_model_config=model_json,
                global_model_weights=global_model_weights,
                num_samples=samples,
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
            if local_run:
                results.append(local_fit(**kwargs))
            else:
                with FuncXExecutor(endp) as fx:
                    tasks.append(fx.submit(local_fit, **kwargs))

        # Retrieve and store the results from the training nodes.
        if not local_run:
            for t in tasks:
                results.append(t.result())

        # Extract model updates from endpoints and then aggregate them to update the global model.
        model_weights = [res["model_weights"] for res in results]
        weights = np.array([res["samples_count"] for res in results])
        weights = weights / weights.sum(0)
        average_weights = np.average(model_weights, weights=weights, axis=0)
        global_model.set_weights(average_weights)
        logging.info(f"Epoch {i}, Trained Federated Model")

        # Evaluate the global model, if all the necessary parameters are given.
        if all([x_test is not None, y_test is not None]):
            test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)
            if not silent:
                logging.info(f"Test loss: {test_loss}  |  Test accuracy: {test_acc}")

        # If `time_interval` is supplied, wait for `time_interval` seconds
        if time_interval > 0:
            time.sleep(time_interval)

    return global_model
