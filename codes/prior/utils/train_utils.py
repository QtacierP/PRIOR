import logging
from pytorch_lightning import seed_everything, Trainer
import flash
import os
import hydra
from omegaconf import DictConfig, OmegaConf, MISSING
import pytorch_lightning as pl
from typing import Dict, Any, List
from flash.image import ObjectDetectionData

log = logging.getLogger(__name__)


def setup_training(config):
    if "seed" in config:
        seed_everything(config.seed)
    if config.debug:
        log.info(f"Running in debug mode! <{config.debug}>")
        config.num_workers = 0
        config.trainer.fast_dev_run = True
        if 'precision' in config.trainer:
            config.trainer.precision = 32
        os.environ['WANDB_MODE'] = 'dryrun'

def init_callbacks(config, callbacks=None):
    if callbacks is None:
        callbacks = []
    else:
        assert isinstance(callbacks, list)
    if config.callbacks is None:
        return callbacks
    for _, cb_conf in config.callbacks.items():
        if cb_conf is None:
            continue
        elif "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
        else:
            raise ValueError(cb_conf)
    return callbacks


def init_loggers(config, loggers=None):
    if loggers is None:
        loggers = []
    else:
        assert isinstance(loggers, list)
    if config.logger is None:
        return loggers
    for _, lg_conf in config.logger.items():
        if lg_conf is None:
            continue
        elif "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))
        else:
            continue
    return loggers


def init_pretrain_model(config, image_encoder=None, text_encoder=None):
    log.info(f"Instantiating pretrain_model <{config.pretrain_model._target_}>")
    pretrain_model = hydra.utils.instantiate(config.pretrain_model)
    return  pretrain_model

def init_downstream_model(config, index=0):
    if config.task == 'segmentation':
        if 'fold' in config.segmentation_model.train_dataset.keys():
            config.segmentation_model.train_dataset.fold = index
            config.segmentation_model.validation_dataset.fold = index
            config.segmentation_model.test_dataset.fold = index
        log.info(f"Instantiating segmentation_model <{config.segmentation_model._target_}>")
        model = hydra.utils.instantiate(config.segmentation_model)
    elif config.task == 'classification':
        if 'fold' in config.classification_model.train_dataset.keys():
            config.classification_model.train_dataset.fold = index
            config.classification_model.validation_dataset.fold = index
            config.classification_model.test_dataset.fold = index
        log.info(f"Instantiating classification_model <{config.classification_model._target_}>")
        model = hydra.utils.instantiate(config.classification_model)
    elif config.task == 'detection':
        if 'fold' in config.detection_model.train_dataset.keys():
            config.detection_model.train_dataset.fold = index
            config.detection_model.validation_dataset.fold = index
            config.detection_model.test_dataset.fold = index
        log.info(f"Instantiating detection_model <{config.detection_model._target_}>")
        model = hydra.utils.instantiate(config.detection_model)
    elif config.task == 'zero_shot_classification':
        log.info(f"Instantiating classification_model <{config.classification_model._target_}>")
        model = hydra.utils.instantiate(config.classification_model)
    return model


def init_trainer(model, config, callbacks=None, logger=None):
    if callbacks is None:
        callbacks = init_callbacks(config)
    else:
        assert isinstance(callbacks, list)
    logger = init_loggers(config, logger)
    log.info(f"Instantiating Trainer ")
    if model.task == 'detection':
        trainer = flash.Trainer(callbacks=callbacks, logger=logger, **config.trainer)
    else:
        trainer = Trainer(callbacks=callbacks, logger=logger, **config.trainer)
    log_hyperparameters(config, model, trainer, logger)
    return  trainer

def init_data(model):
    data_module = ObjectDetectionData.from_files(train_files=model.train_dataset.image_list, train_bboxes=model.train_dataset.box_list, train_targets=model.train_dataset.target_list, val_files=model.validation_dataset.image_list, val_bboxes=model.validation_dataset.box_list, val_targets=model.validation_dataset.target_list, test_files=model.test_dataset.image_list, test_bboxes=model.test_dataset.box_list, test_targets=model.test_dataset.target_list, batch_size=model.batch_size, train_transform=model.train_dataset.image_transform, val_transform=model.validation_dataset.image_transform, test_transform=model.test_dataset.image_transform, input_cls=model.train_dataset.io)
    return data_module
    

def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        trainer: pl.Trainer,
        logger: List[pl.loggers.LightningLoggerBase],
):
    hparams = OmegaConf.to_container(config)
    # save number of model parameters
    params = {}
    params["params_total"] = sum(p.numel() for p in model.parameters())
    params["params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    params["params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    hparams['params'] = params

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just to prevent trainer logging hparams of model as we manage it ourselves)
    for lg in logger:
        lg.log_hyperparams = lambda x: None