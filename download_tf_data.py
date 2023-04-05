import argparse
import numpy as np
import requests
import ssl

from pathlib import Path
from tensorflow import keras

requests.packages.urllib3.disable_warnings()


def main(args: argparse.Namespace):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context

    dataset_map = {
        "mnist": keras.datasets.mnist,
        "fashion_mnist": keras.datasets.fashion_mnist,
    }

    data_dir = Path(args.dir) \
        if (args.dir is not None) \
        else Path.home() / "data"
    data_dir = (data_dir / args.name).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    filenames = ("x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy")
    if args.force or not all([(data_dir / fn).exists() for fn in filenames]):
        # Save the numpy arrays in the specified directory.
        (x_train, y_train), (x_test, y_test) = dataset_map[args.name].load_data()
        np.save(data_dir / "x_train.npy", x_train)
        np.save(data_dir / "y_train.npy", y_train)
        np.save(data_dir / "x_test.npy", x_test)
        np.save(data_dir / "y_test.npy", y_test)
    else:
        print(
            f"-> Files are already downloaded. Run again and pass `--force` if you "
            f"wish to still download them again."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default="mnist", choices=["mnist", "fashion_mnist"],
                        help="Name of the dataset you want to download.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="The code will first check if the filenames are already downloaded. Pass `--force` "
                             "to make sure it re-downloads the data-set from scratch.")
    args = parser.parse_args()
    main(args)
