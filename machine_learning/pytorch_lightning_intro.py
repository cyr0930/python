import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning.metrics.functional import accuracy


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = f'{os.getcwd()}/tmp'

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        MNIST(self.data_path, train=True, download=True)
        MNIST(self.data_path, train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if stage == 'fit':
            mnist_train = MNIST(self.data_path, train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        elif stage == 'test':
            self.mnist_test = MNIST(self.data_path, train=False, transform=transform)

    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
        return mnist_test


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(28 * 28, 128)
        self.layer2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self._process_batch(batch)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        result.log('train_acc', acc, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        loss, acc = self._process_batch(batch)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        result.log('val_acc', acc)
        return result

    def test_step(self, batch, batch_idx):
        loss, acc = self._process_batch(batch)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        result.log('test_acc', acc)
        return result

    def _process_batch(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc


if __name__ == "__main__":
    model = LitModel()
    dm = MNISTDataModule()
    trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=2, gpus=1)
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)
