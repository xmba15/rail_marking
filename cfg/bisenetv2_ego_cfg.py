#!/usr/bin/env python


__all__ = ["BiSeNetV2EgoConfig"]


class BiSeNetV2EgoConfig:
    img_height = 512
    img_width = 1024
    num_classes = 2
    batch_size = 12
    batch_multiplier = 5
    ohem_ce_loss_thresh = 0.7
    num_epochs = 300
    len_epoch = None
    lr_rate = 2.5e-2
    momentum = 0.9
    weight_decay = 5e-4
    burn_in = 1000
    gamma = 0.1
    num_workers = 4
    random_seed = 12
    save_period = 1
    print_after_batch_num = 10
    dataset_name_base = "bisenetv2"
