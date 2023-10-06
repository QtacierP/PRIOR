
import hydra
from omegaconf import OmegaConf
c = OmegaConf
import wandb
from pytorch_lightning import Trainer
from torchvision import transforms
import sys
import os
import torch
import logging
import numpy as np

log = logging.getLogger(__name__)

sys.path.append(os.getcwd())

from prior.utils.train_utils import setup_training, init_trainer, init_downstream_model, init_data
from prior.utils.wandb_utils import finish_run


@hydra.main(config_path="../../configs", config_name="base_finetune")
def finetune(config: OmegaConf):
        if config.task == 'zero_shot_classification':
            total_results = zero_shot(config)
        else:
            raise NotImplementedError
        log.info(f"Total Results: {total_results}")


def zero_shot(config, index=0):
    setup_training(config)
    model = init_downstream_model(config, index=index)
    trainer = init_trainer(model, config)
    results = trainer.test(model)
    torch.cuda.empty_cache()
    #print(convert_result_to_dict(results))
    return trainer, results[0]



if __name__ == '__main__':
    finetune()
    