def local_fit(
        endpoint_name,
        json_model_config,
        global_model_weights,
        train_indices,
        epochs=1,
        keras_dataset="mnist",
        preprocess=True,
        data_dir=None,
        input_shape=(32, 28, 28, 1),
        optimizer="adam",
        metrics=None,
        use_proxystore: bool = False
):
    ###############################################################################################
    # Import all the dependencies required for funcX, TensorFlow, and ProxyStore
    ###############################################################################################
    import numpy as np
    import os
    import sys
    import time
    import pickle
    from funcx.serialize import FuncXSerializer
    from tensorflow import keras
    from pathlib import Path
    from proxystore.proxy import is_resolved, Proxy, resolve, extract
    from proxystore.store import get_store, Store

    FUNCX_SIZE_LIMIT = 10485760
    MODEL_TRAIN_EVAL = True
    fxs = FuncXSerializer()
    if metrics is None:
        metrics = ["accuracy"]

    def write_status(text: str) -> None:
        os.system(f"echo \"{text}\" > ~/status.txt")

    ###############################################################################################
    # Load the datasets from the provided path directory.
    ###############################################################################################

    write_status("Preparing to load the training data.")
    legal_datasets = ["mnist", "fashion_mnist"]
    if keras_dataset not in legal_datasets:
        raise Exception(f"Please select one of the built-in Keras datasets: {legal_datasets}.")
    if data_dir is None:
        data_dir = Path.home() / "data" / keras_dataset
    x_train = np.load(data_dir / "x_train.npy")[train_indices]
    y_train = np.load(data_dir / "y_train.npy")[train_indices]
    if preprocess and keras_dataset in legal_datasets:
        num_classes = 100 if keras_dataset == "cifar100" else 10
        x_train = x_train.astype("float32") / 255
        if x_train.shape[-1] not in [1, 3]:
            x_train = np.expand_dims(x_train, -1)
        y_train = keras.utils.to_categorical(y_train, num_classes)
    write_status("Data is loaded and preprocessed (if specified).")

    ###############################################################################################
    # Resolve the remote data store from the global model weights if ProxyStore is being used.
    ###############################################################################################

    write_status("Starting to resolve store, if any.")
    if use_proxystore:
        write_status("Before the isinstance(*) calls")
        assert isinstance(global_model_weights, Proxy)
        assert not is_resolved(global_model_weights)
        store = get_store(global_model_weights)
        if store is None:
            raise RuntimeError("Cannot find ProxyStore backend to use.")
    else:
        store = None
    write_status("Finished resolution of the store.")

    ###############################################################################################
    # Perform the local training for the federated learning process then return the params.
    ###############################################################################################

    write_status(f"Building/compiling the model from the JSON config, then fitting.")
    if use_proxystore:
        # Needed to not throw an error in `keras.models.model_from_json(*)`.
        json_model_config = extract(json_model_config)
    model = keras.models.model_from_json(json_model_config)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=metrics
    )

    write_status("Before setting the local model to use the global weights")
    model.set_weights(global_model_weights)
    if MODEL_TRAIN_EVAL:
        history = model.fit(x_train, y_train, epochs=epochs).history
    model_weights = np.asarray(model.get_weights(), dtype=object)

    write_status("Preparing to transfer data back to the controller.")
    data = {
        "endpoint_name": endpoint_name,
        "model_weights": store.proxy(model_weights, evict=False) if use_proxystore else model_weights,
        "samples_count": x_train.shape[0],
        "accuracy": float(0),
        "loss": float(0),
        "num_data_samples": len(train_indices),
        "data_transfer_size": float(0),
        "start_transfer_time": time.time()
    }
    if MODEL_TRAIN_EVAL:
        data["accuracy"] = history["accuracy"]
        data["loss"] = history["loss"]
    data["data_transfer_size"] = sys.getsizeof(fxs.serialize(data))

    time.sleep(2)

    if not use_proxystore and sys.getsizeof(fxs.serialize(data)) > FUNCX_SIZE_LIMIT:
        write_status("funcX size limit EXCEEDED for return")
        return None
    else:
        write_status("funcX size limit RESPECTED for return")
        return data
