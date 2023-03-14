import argparse
import os
import yaml

from models.congen_model import ConGenModel
from models.uncond_model import UncondModel
<<<<<<< HEAD
from models.uncond_model_attention import UncondModelAttention
=======
from models.attn_uncond_model import AttentionUncondModel
from models.attn_congen_model import AttnConGenModel
>>>>>>> c36bd9a3ed27244f2361a13996382bca3f2c4e59
from dataset import HandwrittingDataModule

import torch.cuda

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import pdb

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v')
<<<<<<< HEAD
    parser.add_argument('--model', '-m', default='uncond_attention', choices=["cond", "uncond", "uncond_attention"])
    # parser.add_argument('--model', '-m', default='uncond', choices=["cond", "uncond"])
=======
    parser.add_argument('--model', '-m', default='cond', choices=["cond", "uncond", "attn_uncond"])
>>>>>>> c36bd9a3ed27244f2361a13996382bca3f2c4e59
    parser.add_argument('--weights_path', '-w', default=None)

    args = parser.parse_args()

    # Read config file
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    training_params = params['TrainingParams']

    # Initialize data module
    data_module = HandwrittingDataModule(
        batch_size=training_params['batch_size'],
        num_workers=training_params['num_workers'],
        model=args.model
    )
    data_module.setup(stage='fit')

    # Initialize new model and setup data module
    model = None
    if args.model == 'cond':
        model = ConGenModel()
    elif args.model == "uncond":
        model = UncondModel()
<<<<<<< HEAD
    elif args.model == "uncond_attention":
        model = UncondModelAttention()
=======
    elif args.model == "attn_uncond":
        model = AttentionUncondModel()
    elif args.model == "attn_cond":
        model = AttnConGenModel()
>>>>>>> c36bd9a3ed27244f2361a13996382bca3f2c4e59

    if args.weights_path is not None:
        model = model.load_from_checkpoint(args.weights_path)

    # Loggers and checkpoints
    version = args.version
    logger = TensorBoardLogger('.', version=version)
    model_ckpt = ModelCheckpoint(dirpath=f'lightning_logs/{version}/checkpoints',
                                 save_top_k=2,
                                 monitor='loss_train',
                                 mode='min',
                                 save_weights_only=True)
    lr_monitor = LearningRateMonitor()

    # Trainer
    trainer = Trainer(accelerator='auto',
                      devices=1 if torch.cuda.is_available() else None,
                      max_epochs=60,
                      num_sanity_val_steps=0,
                      limit_val_batches=0.0,
                      callbacks=[model_ckpt, lr_monitor],
                      logger=logger,
                      gradient_clip_val=training_params["clip"])
    
    trainer.fit(model, data_module)
