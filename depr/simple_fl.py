import torch
import torch.nn.functional as F
import torchvision
import os
import pytorch_lightning as pl

from funcx import FuncXExecutor
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torchmetrics.functional import accuracy

from depr.data import load_federated_cifar10
from endpoint import Endpoint, LocalFitIn

PATH_DATASETS = os.environ.get("PATH_DATASETS", "../data/")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)


class LitResnet(pl.LightningModule):
    def __init__(self, lr: float = 0.05):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(pretrained=False, num_classes=10)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task='multiclass')
        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


def fed_avg(
        state_dicts: list[dict[str, torch.Tensor]],
        num_samples: list[int]
) -> dict[str, torch.tensor]:
    total_samples = sum(num_samples)
    weights = [s / total_samples for s in num_samples]

    avg_state = {}
    for state, weight in zip(state_dicts, weights):
        for layer, params in state.items():
            params = torch.clone(params.detach())
            if layer not in avg_state:
                avg_state[layer] = params * weight
            else:
                avg_state[layer] += params * weight

    return avg_state


def local_fit(
        module: pl.LightningModule,
        endpoint_id: str,
        train_indices: list[int],
        test_indices: list[int],
        **hyperparams
) -> LocalFitIn:
    batch_size = hyperparams.get('batch_size', 64)

    train_loader, test_loader = load_federated_cifar10(train_indices, test_indices, batch_size)

    trainer = pl.Trainer(
        max_epochs=hyperparams.get('max_epochs', 3),
        accelerator='auto',
        devices=1 if any([torch.cuda.is_available(), torch.backends.mps.is_available()]) else None,
        logger=CSVLogger(save_dir='logs/'),
        callbacks=[LearningRateMonitor(logging_interval='step'),
                   TQDMProgressBar(refresh_rate=10)]
    )
    trainer.fit(module, train_dataloader)

    return LocalFitIn(endpoint_id=endpoint_id, module=module, num_samples=len(train_dataloader))


def federated_fit(
        endpoints: list[Endpoint],
        module: pl.LightningModule,
        num_rounds: int
):
    endpoint_ids = [endp.idx for endp in endpoints]
    if len(endpoint_ids) == len(set(endpoint_ids)):
        raise ValueError('Duplicate endpoints are currently not supported.')

    # Begin the federated training rounds.
    for round_no in range(num_rounds):
        # Local training at each endpoint.
        results: list[LocalFitIn] = []
        futures = []
        for endp in endpoints:
            kwargs = {'client_id': endp.idx, 'module': module}
            if endp.idx.startswith('local'):
                results.append(local_fit(**kwargs))
            else:
                with FuncXExecutor(endpoint_id=endp.idx) as fxe:
                    fut = fxe.submit(local_fit, **kwargs)
                    futures.append(fut)

        # Retrieve the results from the futures (if any).
        for fut in futures:
            res = fut.result()
            results.append(res)

        # Collect model parameters/weights to perform aggregation.
        client_state_dicts, client_samples = [], []
        for res in results:
            client_state_dicts.append(res.module.state_dict())
            client_samples.append(res.num_samples)


def main() -> None:
    indices = []
    endpoints = [
        Endpoint(f'local:{i}', indices[0])
        for i in range(len(indices))
    ]
    # Initialize the "global" PyTorch Lightning Module for training.
    module = pl.LightningModule()
    federated_fit(endpoints, module, 10)


if __name__ == '__main__':
    # samples = [1, 10]
    # state_dicts = [
    #     {'input': torch.tensor([[1, 2, 3], [4, 5, 6]]),
    #      'output': torch.tensor([7, 8, 9])},
    #     {'input': torch.tensor([[9, 8, 7], [6, 5, 4]]),
    #      'output': torch.tensor([3, 2, 1])}
    # ]
    #
    # start = time.perf_counter()
    # global_params = fed_avg(state_dicts, samples)
    # wall_time = time.perf_counter() - start
    # print(f'{global_params}\n'
    #       f'Time taken: {wall_time} seconds.')

    train_indices = list(range(100))
    test_indices = list(range(10))

    train_loader, test_loader = load_federated_cifar10(train_indices, test_indices, batch_size=4)
    print(f'Number of batches in Train-Loader: {len(train_loader)}\n'
          f'Number of batches in Test-Loader:  {len(test_loader)}')
