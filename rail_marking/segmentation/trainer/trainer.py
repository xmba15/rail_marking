#!/usr/bin/env python
import logging
import torch
from ...core.trainer import TrainerBase


__all__ = ["BiSeNetV2Trainer"]


class BiSeNetV2Trainer(TrainerBase):
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

        TrainerBase.__init__(
            self,
            model=model,
            criterion=criterion,
            metric_func=metric_func,
            optimizer=optimizer,
            data_loaders_dict=data_loaders_dict,
            config=config,
            scheduler=scheduler,
            device=device,
            logger=logger,
        )

    def _train_epoch(self, epoch):
        self._model.train()

        epoch_train_loss = 0.0
        count = self._batch_multiplier
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(self._train_data_loader):

            data = data.to(self._device)
            target = target.to(self._device)

            if count == 0:
                self._optimizer.step()
                self._optimizer.zero_grad()
                count = self._batch_multiplier

            with torch.set_grad_enabled(True):
                logits, *logits_aux = self._model(data)
                loss_pre = self._criterion(logits, target)
                loss_aux = [self._criterion(lgt, target) for lgt in logits_aux]

                train_loss = loss_pre + sum(loss_aux)

                total_loss = train_loss / self._batch_multiplier
                total_loss.backward()
                count -= 1

                running_loss += train_loss
                if (batch_idx + 1) % self._print_after_batch_num == 0:
                    # logging.info(
                    #     "\n epoch: {}/{} | | iter: {}/{} | | [Losses: total: {}".format(
                    #         epoch,
                    #         self._num_epochs,
                    #         batch_idx,
                    #         len(self._train_data_loader),
                    #         running_loss,
                    #     )
                    # )
                    print(
                        "\n epoch: {}/{} | | iter: {}/{} | | [Losses: total: {}".format(
                            epoch,
                            self._num_epochs,
                            batch_idx,
                            len(self._train_data_loader),
                            running_loss,
                        )
                    )
                    running_loss = 0.0

                epoch_train_loss += train_loss.item()

            if batch_idx == self._len_epoch:
                break

        if self._do_validation:
            epoch_val_loss = self._valid_epoch(epoch)
            print(
                "\n epoch: {}/{} | | [Val Losses: total: {}".format(
                    epoch,
                    self._num_epochs,
                    epoch_val_loss,
                )
            )

        if self._scheduler is not None:
            self._scheduler.step()

        return (
            epoch_train_loss / self._num_train_imgs,
            epoch_val_loss / self._num_val_imgs,
        )

    def _valid_epoch(self, epoch):
        print("start validation...")
        self._model.eval()

        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self._val_data_loader):
                data = data.to(self._device)
                target = target.to(self._device)

                logits = self._model(data)
                val_loss = self._criterion(logits, target)

                epoch_val_loss += val_loss.item()

        return epoch_val_loss
