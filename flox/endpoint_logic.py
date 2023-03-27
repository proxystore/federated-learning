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
        # Arguments for ProxyStore Store object
        store_kind='Redis',
        store_name=None,
        store_hostname=None,
        store_port=None,
):
    # Import all the dependencies required for funcX, TensorFlow, and ProxyStore
    import numpy as np
    from tensorflow import keras
    from time import perf_counter
    from proxystore.connectors.redis import RedisConnector
    from proxystore.store import get_store, register_store, Store

    if metrics is None:
        metrics = ["accuracy"]

    # retrieve (and optionally process) the data
    AVAILABLE_DATASETS = [
        'mnist', 'fashion_mnist', 'cifar10', 'cifar100',
        'imdb', 'reuters', 'boston_housing'
    ]
    DATASET_MAP = {
        'mnist': keras.datasets.mnist,
        'fashion_mnist': keras.datasets.fashion_mnist,
        'cifar10': keras.datasets.cifar10,
        'cifar100': keras.datasets.cifar100,
        'imdb': keras.datasets.imdb,
        'reuters': keras.datasets.reuters,
        'boston_housing': keras.datasets.boston_housing
    }
    IMG_DATASETS = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']

    ####################################################################################################################
    ####################################################################################################################

    # check if the Keras dataset exists
    if keras_dataset not in AVAILABLE_DATASETS:
        raise Exception(f"Please select one of the built-in Keras datasets: {AVAILABLE_DATASETS}")

    # load the data
    (x_train, y_train), _ = DATASET_MAP[keras_dataset].load_data()

    # take a random set of images
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    if preprocess and keras_dataset in IMG_DATASETS:
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
    print(f'{type(history)=}\n{history.history.keys()}')
    model_weights = model.get_weights()
    np_model_weights = np.asarray(model_weights, dtype=object)

    # Return the updated weights and number of samples the model was trained on
    store_args = [store_name, store_port, store_hostname]
    data = {
        "endpoint_id": endpoint_id,
        "model_weights": np_model_weights,
        "samples_count": x_train.shape[0],
        "accuracy": history.history["accuracy"],
        "loss": history.history["loss"],
        "num_data_samples": len(train_indices)
    }
    if all([arg is None for arg in store_args]):
        data["time_before_transfer"] = perf_counter()
        return data
    elif all([arg is not None for arg in store_args]):
        data["time_before_transfer"] = perf_counter()
        store = Store(
            name=store_name,
            connector=RedisConnector(hostname=store_hostname, port=store_port),
            metrics=True
        )
        proxy = store.proxy(data)
        return proxy
    else:
        raise ValueError(f'Must either have ALL store-based params (i.e., {store_args}) '
                         f'be None or you must give them ALL a non-None value.')
