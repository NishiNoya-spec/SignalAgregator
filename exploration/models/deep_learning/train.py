from collections import defaultdict
from glob import glob

import numpy as np
import torch

# import torch.nn.functional as F
import torch.optim as optim

# Подготовка данных
from torch.utils.data import DataLoader
from tqdm import tqdm

from exploration.models.deep_learning.constants import (
    INPUT_SIZE,
    OUTPUT_SIZE,
    PATH_TO_TRAIN_DATA,
    PATH_TO_VAL_DATA,
    THRESHOLD,
)
from exploration.models.deep_learning.dataset import BilletsDataset
from exploration.models.deep_learning.loss import ModifiedMSELoss
from exploration.models.deep_learning.model import TorsionUNet

torch.cuda.empty_cache()


def get_iou(outputs, labels):
    interception = np.sum(outputs * labels)
    union = np.sum(outputs + labels) - interception
    return interception / union


PATH_TO_MODEL = r"best.pth"


class Train:

    def __init__(
        self, num_epochs: int, lr: float, batch: int, pretrained_model: bool
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = TorsionUNet(INPUT_SIZE, OUTPUT_SIZE)
        if pretrained_model:
            self.model.load_state_dict(torch.load(PATH_TO_MODEL, ))

        self.train_dataloader = self._get_dataloader(PATH_TO_TRAIN_DATA, batch)
        self.val_dataloader = self._get_dataloader(PATH_TO_VAL_DATA, batch)

        self.criterion = ModifiedMSELoss(
            THRESHOLD, 4, 3, 6, 50
        )    # torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.num_epochs = num_epochs
        self.best_loss = 0
        self.model.to(self.device)

    def _get_dataloader(self, data_path, batch):
        dataset = BilletsDataset(glob(data_path))
        dataloader = DataLoader(
            dataset, batch_size=batch, shuffle=True, num_workers=4
        )
        return dataloader

    def start(self):
        print("Start training....")
        for epoch in range(self.num_epochs):
            print(f"epoch {epoch}")
            self._train(epoch)
            loss = self._evaluate(epoch)
            if epoch == 0 or self.best_loss > loss:
                print("Saving model....")
                self.best_loss = loss
                torch.save(self.model.state_dict(), "best.pth")
            torch.save(self.model.state_dict(), "last.pth")

    def _train(self, epoch):
        self.model.train()    # Переключение модели в режим обучения
        log_data = defaultdict(list)
        pbar = tqdm(self.train_dataloader)
        for inputs, labels in pbar:    # Перебор тренировочных данных
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()    # Обнуление градиентов

            # Прямой проход
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels, epoch)

            # Обратный проход и оптимизация
            loss.backward()
            self.optimizer.step()

            log_data = self._update_regression_metrics(
                outputs, labels, loss, log_data
            )
            pbar.set_postfix(
                {key: np.mean(val)
                 for key, val in log_data.items()}
            )

    def _evaluate(self, epoch):
        # Оценка модели
        self.model.eval()    # Переключение модели в режим оценки
        with torch.no_grad():
            log_data = defaultdict(list)
            pbar = tqdm(self.val_dataloader)
            for inputs, labels in pbar:    # Перебор тестовых данных
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels, epoch)
                log_data = self._update_regression_metrics(
                    outputs, labels, loss, log_data
                )
                pbar.set_postfix(
                    {key: np.mean(val)
                     for key, val in log_data.items()}
                )

        return loss.item()

    def _update_regression_metrics(self, loss, log_data):

        log_data["loss"].append(loss.item())
        return log_data


if __name__ == "__main__":
    trainer = Train(
        num_epochs=300,
        lr=0.001,
        batch=512,
        pretrained_model=False,
    )
    trainer.start()
