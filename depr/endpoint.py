from typing import Optional

import pytorch_lightning as pl

from dataclasses import dataclass, field

from numpy.random import RandomState


@dataclass
class Endpoint:
    idx: str
    train_indices: list[int] = field(hash=False)
    test_indices: list[int] = field(hash=False)


@dataclass
class LocalFitIn:
    endpoint_id: str
    module: pl.LightningModule
    num_samples: int


def init_endpoint_data_indices(
        endpoint_ids: list[str],
        num_training_samples: int,
        num_testing_samples: int,
        overlapping: bool = False,
        random_state: Optional[RandomState] = None
):
    if random_state is None:
        random_state = RandomState()

    universe_training_indices = set(range(num_training_samples))
    universe_testing_indices = set(range(num_testing_samples))

    indices = {
        endp: {'train': [], 'test': []}
        for endp in endpoint_ids
    }
