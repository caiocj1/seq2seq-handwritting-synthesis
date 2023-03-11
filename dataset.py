import numpy as np
import os
import yaml
from yaml import SafeLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn


class StrokeDataset(Dataset):
    def __init__(self):
        super(StrokeDataset, self).__init__()

        self.strokes = np.load('./data/strokes.npy', allow_pickle=True, encoding='latin1')
        with open('./data/sentences.txt') as f:
            self.texts = f.readlines()
        self.texts = [a.split('\n')[0] for a in self.texts]

    def __getitem__(self, item):
        return self.strokes[item], self.texts[item]

    def __len__(self):
        return len(self.texts)


class HandwrittingDataModule(LightningDataModule):
    def __init__(self,
            batch_size: int = 32,
            num_workers: int = 0,
            model: str = None):
        super().__init__()

        self.model = model

        # Save hyperparemeters
        self.save_hyperparameters(logger=False)

        # Read config file
        self.read_config()

    def read_config(self):
        """
        Read configuration file with hyperparameters.
        :return: None
        """
        config_path = os.path.join(os.getcwd(), 'config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        dataset_params = params["DatasetParams"]

        self.min_seq = dataset_params["min_seq"]
        self.max_seq = dataset_params["max_seq"] if self.model == "cond" else dataset_params["max_seq_uncond"]
        self.max_text_len = dataset_params["max_text_len"]

    def setup(self, stage: str = None):
        """
        Build data dictionaries for training or prediction.
        :param stage: 'fit' for training, 'predict' for prediction
        :return: None
        """
        if stage == 'fit':
            train_dataset = StrokeDataset()
            val_dataset = StrokeDataset()

            self.data_train, self.data_val = train_dataset, val_dataset

        elif stage == 'predict':
            predict_dict = None

            self.data_predict = predict_dict

    def train_dataloader(self):
        """
        Uses train dictionary (output of format_X) to return train DataLoader, that will be fed to pytorch lightning's
        Trainer.
        :return: train DataLoader
        """
        return DataLoader(dataset=self.data_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          collate_fn=self.collate_fn(self.model),
                          shuffle=True)

    def val_dataloader(self):
        """
        Uses val dictionary (output of format_X) to return val DataLoader, that will be fed to pytorch lightning's
        Trainer.
        :return: train DataLoader
        """
        return DataLoader(dataset=self.data_val,
                         batch_size=self.hparams.batch_size,
                         num_workers=self.hparams.num_workers,
                         collate_fn=self.collate_fn(self.model),
                         shuffle=False)

    def predict_dataloader(self):
        """
        Uses predict dictionary (output of format_X) to return predict DataLoader, that will be fed to pytorch
        lightning's Trainer.
        :return: predict DataLoader
        """
        return DataLoader(dataset=self.data_predict,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def collate_fn(self, model):
        if model == "cond":
            return self.get_strokes_text
        elif model == "uncond":
            return self.get_data_uncond

    def get_data_uncond(self, batch):
        # ind=0, batch_size=1, max_seq=400

        batch_size = len(batch)

        big_x, big_y = [], []
        # print ('here')

        for k in range(batch_size):
            X = batch[k][0]
            if len(X) < self.max_seq:
                continue
            # print 'here'
            halt = int(len(X) / self.max_seq) + 1
            # print 'here',len(X),'halt',halt
            count = 0
            for j in range(0, len(X), self.max_seq):
                y = []
                x = [[0, 0, 0]]
                if count == halt - 1:
                    for i in range(len(X) - self.max_seq, len(X)):
                        y.append(X[i])
                    y.append(X[i])
                    x.extend(X[len(X) - self.max_seq:])
                    big_x.append(x)
                    big_y.append(y)
                    continue
                else:
                    for i in range(j, min(j + self.max_seq, len(X))):
                        y.append(X[i])
                    y.append(X[i])
                    x.extend(X[j:min(j + self.max_seq, len(X))])
                    y = np.array(y)
                    big_x.append(x)
                    big_y.append(y)
                count += 1
        X = torch.tensor(np.array(big_x))
        y = torch.tensor(np.array(big_y))
        return X, y

    def get_strokes_text(self, batch):
        big_x, big_y, big_text = [], [], []
        stroke_mask, text_mask, len_text = [], [], []

        batch_size = len(batch)

        k = 0
        count = 0

        while (count < batch_size):
            X = batch[k][0]
            mask = np.ones(self.max_seq)

            # if len(X) < self.min_seq:
            #     k += 1
            #     # k = k-2
            #     continue

            x = []
            for i in range(min(len(X), self.max_seq)):
                x.append(X[i].tolist())

            if len(X) < self.max_seq:
                for i in range(self.max_seq - len(X)):
                    x.append([0, 0, 0])

                mask[len(X):] = 0
            stroke_mask.append(mask)
            X = np.array(x)

            y = []
            for i in range(1, len(X)):
                y.append(X[i])
            y.append(X[len(X) - 1])
            y = np.array(y)

            char_list = ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,."\'?-!'
            char_to_code = {}
            code_to_char = {}
            c = 0
            for ch in char_list:
                char_to_code[ch] = c
                code_to_char[c] = ch
                c += 1
            text = batch[k][1]
            text = text[0:min(self.max_text_len, len(text))]

            vectors = np.zeros((self.max_text_len, len(char_to_code) + 1))

            mask = np.ones(self.max_text_len)
            for p, q in enumerate(text):
                try:
                    vectors[p][char_to_code[q]] = 1
                except:
                    vectors[p][-1] = 1
                    continue

            if len(text) < self.max_text_len:
                mask[len(text):] = 0
            text_mask.append(mask)
            len_text.append(len(text))

            big_x.append(X)
            big_y.append(y)
            big_text.append(vectors)

            k += 1
            count += 1

        X = torch.tensor(np.array(big_x))
        y = torch.tensor(np.array(big_y))
        text = torch.tensor(np.array(big_text))

        stroke_mask = torch.tensor(np.array(stroke_mask))
        text_mask = torch.tensor(np.array(text_mask))

        len_text = torch.tensor(np.array(len_text))

        return [X, y, text], [stroke_mask, text_mask], len_text, char_to_code, code_to_char