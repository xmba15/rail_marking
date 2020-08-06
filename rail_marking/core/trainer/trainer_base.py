#!/usr/bin/env python
import logging
import os
import torch
import abc


__all__ = ["TrainerBase"]


class TrainerBase:
    def __init__(
        self,
        model,
        criterion,
        metric_func,
        optimizer,
        data_loaders_dict,
        config,
        scheduler=None,
        device=None,
        logger=None,
    ):
        self._model = model
        self._criterion = criterion
        self._metric_func = metric_func
        self._optimizer = optimizer

        self._logger = logger

        if device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

        self._model = self._model.to(self._device)

        self._start_epoch = 1

        self._train_data_loader = data_loaders_dict["train"]
        self._val_data_loader = data_loaders_dict["val"]
        self._num_train_imgs = len(self._train_data_loader.dataset)
        self._num_val_imgs = len(self._val_data_loader.dataset)

        self._config = config
        self._batch_multiplier = self._config.batch_multiplier
        self._checkpoint_dir = self._config.saved_model_path
        self._num_epochs = self._config.num_epochs
        self._save_period = self._config.save_period
        self._dataset_name_base = self._config.dataset_name_base
        self._print_after_batch_num = self._config.print_after_batch_num

        if self._config.len_epoch is None:
            self._len_epoch = len(self._train_data_loader)
        else:
            self.train_data_loader = TrainerBase.inf_loop(self._train_data_loader)
            self._len_epoch = self._config.len_epoch
        self._do_validation = self._val_data_loader is not None
        self._scheduler = scheduler

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self._model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        output_file = "checkpoint_{}_epoch_{}.pth".format(arch, epoch)
        if self._dataset_name_base and isinstance(self._dataset_name_base, str) and self._dataset_name_base != "":
            output_file = "{}_{}".format(self._dataset_name_base, output_file)

        filename = os.path.join(self._checkpoint_dir, output_file)
        torch.save(state, filename)

    def resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)

        checkpoint = torch.load(resume_path)
        self._start_epoch = checkpoint["epoch"] + 1

        self._model.load_state_dict(checkpoint["state_dict"])

        self._optimizer.load_state_dict(checkpoint["optimizer"])

    def train(self):
        logging.info("========================================")
        logging.info("Start training {}".format(type(self._model).__name__))
        logging.info("========================================")
        logs = []

        for epoch in range(self._start_epoch, self._num_epochs + 1):
            train_loss, val_loss = self._train_epoch(epoch)

            log_epoch = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            logging.info("========================================")
            logging.info("epoch: {}, train_loss: {: .4f}, val_loss: {: .4f}".format(epoch, train_loss, val_loss))
            logging.info("========================================")
            logs.append(log_epoch)
            if self._logger:
                self._logger.add_scalar("train/train_loss", train_loss, epoch)
                self._logger.add_scalar("val/val_loss", val_loss, epoch)

            if (epoch + 1) % self._save_period == 0:
                self._save_checkpoint(epoch, save_best=True)

        return logs

    @staticmethod
    def inf_loop(data_loader):
        from itertools import repeat

        for loader in repeat(data_loader):
            yield from loader
