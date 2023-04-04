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
    import sys
    import pickle
    from tensorflow import keras
    from time import perf_counter
    from proxystore.proxy import is_resolved, Proxy
    from proxystore.store import get_store, Store

    if metrics is None:
        metrics = ["accuracy"]
        
    ###############################################################################################
    # Load the datasets from the the provided path directory.
    ###############################################################################################
    
    from pathlib import Path
    legal_datasets = ["mnist", "fashion_mnist"]
    if keras_dataset not in legal_datasets:
        raise Exception(f"Please select one of the built-in Keras datasets: {legal_datasets}.")
    if data_dir is None:
        data_dir = Path.home() / "data" / keras_dataset
    x_train = np.load(data_dir / "x_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    # x_test = np.load(data_dir / "x_test.npy")
    # y_test = np.load(data_dir / "y_test.npy")

    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    if preprocess and keras_dataset in img_datasets:
        num_classes = 100 if keras_dataset == "cifar100" else 10
        x_train = x_train.astype("float32") / 255
        if x_train.shape[-1] not in [1, 3]:
            x_train = np.expand_dims(x_train, -1)
        y_train = keras.utils.to_categorical(y_train, num_classes)

    ###############################################################################################
    # Resolve the remote data store from the global model weights if ProxyStore is being used.
    ###############################################################################################

    if use_proxystore:
        assert isinstance(global_model_weights, bytes) and isinstance(global_model_weights, Proxy)
        assert not is_resolved(global_model_weights)
        store = get_store(global_model_weights)
        if store is None:
            raise RuntimeError("Cannot find ProxyStore backend to use.")
    else:
        store = None

    ###############################################################################################
    # Perform the local training for the federated learning process then return the params.
    ###############################################################################################
    
    model = keras.models.model_from_json(json_model_config)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=metrics
    )
    model.set_weights(global_model_weights)
    history = model.fit(x_train, y_train, epochs=epochs).history
    model_weights = np.asarray(model.get_weights(), dtype=object)
    # history = {"accuracy": [0.9], "loss": [3.14]}
    # model_weights = global_model_weights

    # Return the updated weights and number of samples the model was trained on
    data = {
        "endpoint_name": endpoint_name,
        "model_weights": model_weights,
        "samples_count": x_train.shape[0],
        "accuracy": history["accuracy"],
        "loss": history["loss"],
        "num_data_samples": len(train_indices),
        "data_transfer_size": float(0),
        "time_before_transfer": perf_counter()
    }
    data["data_transfer_size"] = sys.getsizeof(pickle.dumps(data))
    return store.proxy(data) if use_proxystore else data
