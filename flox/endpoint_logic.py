def local_fit(
        endpoint_id,
        json_model_config,
        global_model_weights,
        train_indices,
        epochs=10,
        keras_dataset="mnist",
        preprocess=True,
        path_dir="/home/pi/datasets",
        x_train_name="mnist_x_train.npy",
        y_train_name="mnist_y_train.npy",
        input_shape=(32, 28, 28, 1),
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=None,
        use_proxystore: bool = False
):
    # Import all the dependencies required for funcX, TensorFlow, and ProxyStore
    import numpy as np
    import sys
    import pickle
    from tensorflow import keras
    from time import perf_counter
    from proxystore.connectors.redis import RedisConnector
    from proxystore.proxy import is_resolved, Proxy
    from proxystore.store import get_store, register_store, Store

    if metrics is None:
        metrics = ["accuracy"]

    # retrieve (and optionally process) the data
    dataset_map = {
        "mnist": keras.datasets.mnist,
        "fashion_mnist": keras.datasets.fashion_mnist,
        "cifar10": keras.datasets.cifar10,
        "cifar100": keras.datasets.cifar100,
        "imdb": keras.datasets.imdb,
        "reuters": keras.datasets.reuters,
        "boston_housing": keras.datasets.boston_housing
    }
    img_datasets = ["mnist", "fashion_mnist", "cifar10", "cifar100"]

    if use_proxystore:
        assert isinstance(global_model_weights, bytes) and isinstance(global_model_weights, Proxy)
        assert not is_resolved(global_model_weights)
        store = get_store(global_model_weights)
        if store is None:
            raise RuntimeError("Cannot find ProxyStore backend to use.")
    else:
        store = None

    ####################################################################################################################
    ####################################################################################################################

    # Load the data with the indices specified for this client.
    if keras_dataset not in dataset_map:
        raise Exception(f"Please select one of the built-in Keras datasets: {list(dataset_map)}")
    (x_train, y_train), _ = dataset_map[keras_dataset].load_data()
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    if preprocess and keras_dataset in img_datasets:
        num_classes = 100 if keras_dataset == "cifar100" else 10
        x_train = x_train.astype("float32") / 255
        if x_train.shape[-1] not in [1, 3]:
            x_train = np.expand_dims(x_train, -1)
        y_train = keras.utils.to_categorical(y_train, num_classes)

    # Create/compile the model, setting its weights. Then set its weights to the global model.
    model = keras.models.model_from_json(json_model_config)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=metrics
    )
    model.set_weights(global_model_weights)

    # Train the model on the local data and extract the weights then convert them to numpy.
    history = model.fit(x_train, y_train, epochs=epochs)
    model_weights = np.asarray(model.get_weights(), dtype=object)

    # Return the updated weights and number of samples the model was trained on
    data = {
        "endpoint_id": endpoint_id,
        "model_weights": model_weights,
        "samples_count": x_train.shape[0],
        "accuracy": history.history["accuracy"],
        "loss": history.history["loss"],
        "num_data_samples": len(train_indices),
        "data_transfer_size": float(0),
        "time_before_transfer": perf_counter()
    }
    data["data_transfer_size"] = sys.getsizeof(pickle.dumps(data))
    return store.proxy(data) if use_proxystore else data
