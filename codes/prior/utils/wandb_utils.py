import glob
import logging
import os

import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only

log = logging.getLogger(__name__)


def get_wandb_logger(trainer: pl.Trainer) -> WandbLogger:
    logger = None
    for lg in trainer.logger:
        if isinstance(lg, WandbLogger):
            logger = lg

    if not logger:
        log.warn(
            "You are using wandb related callback,"
            "but WandbLogger was not found for some reason..."
        )

    return logger

@ rank_zero_only
def finish_run(trainer):
    logger = None
    for lg in trainer.logger:
        if isinstance(lg, WandbLogger):
            logger = lg
    if logger is not None:
        path = logger.experiment.path
        dir = logger.experiment.dir
        logger.experiment.finish()
        run_api = wandb.Api().run(path)
        return run_api, dir
    else:
        return None, None


class UploadCodeToWandbAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    @ rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.use_artifact(code)


class UploadConfigToWandbAsArtifact(Callback):
    def __init__(self, config_dir: str):
        self.config_dir = config_dir

    @ rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        run_config = wandb.Artifact("run-config", type="config")
        for path in glob.glob(os.path.join(self.config_dir, "*.yaml")):
            log.info(f'Uploading config {path}')
            run_config.add_file(path)

        experiment.use_artifact(run_config)


class UploadCheckpointsToWandbAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of training."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @ rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(
                os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True
            ):
                ckpts.add_file(path)

        experiment.use_artifact(ckpts)


class SaveBestCheckpointPathToWandbSummary(Callback):
    def __init__(self, prefix: str = None):
        self.prefix = '' if prefix is None else prefix + '_'

    @ rank_zero_only
    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        logger = get_wandb_logger(trainer=trainer)
        logger.experiment.summary[self.prefix + 'best_checkpoint'] = trainer.checkpoint_callback.best_model_path


class WatchModelWithWandb(Callback):
    """Make WandbLogger watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @ rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)