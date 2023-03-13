import torch
from models.congen_model import ConGenModel
from models.uncond_model import UncondModel
import os
import yaml


def load_pretrained_congen(ckpt_path):
    char_list = ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,."\'?-!'
    char_to_code = {}
    c = 0
    for ch in char_list:
        char_to_code[ch] = c
        c += 1

    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    model_params = params["CongenModelParams"]
    hidden_size = model_params["hidden_size"]

    lr_model = ConGenModel.load_from_checkpoint(ckpt_path)
    lr_model.eval()

    return lr_model, char_to_code, hidden_size

def load_pretrained_uncond(ckpt_path):
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    model_params = params["CongenModelParams"]
    hidden_size = model_params["hidden_size"]

    lr_model = UncondModel.load_from_checkpoint(ckpt_path)
    lr_model.eval()

    return lr_model, hidden_size
