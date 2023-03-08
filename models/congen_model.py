import torch.optim
from pytorch_lightning import LightningModule
import torch.nn as nn
import os
import yaml
from yaml import SafeLoader
from collections import OrderedDict
import torch.functional as F
import numpy as np

class ConGenModel(LightningModule):

    def __init__(self):
        super(ConGenModel, self).__init__()

        # Read config file if needed
        self.read_config()

        # Save hyperparameters
        if self.bi_dir == True:
            self.bi = 2
        else:
            self.bi = 1

        # Define model itself
        self.rnn1 = nn.LSTM(self.input_size + self.char_vec_len,
                            self.hidden_size,
                            self.n_layers,
                            dropout=self.dropout_p,
                            batch_first=True,
                            bidirectional=True)
        self.rnn2 = nn.LSTM(self.hidden_size * self.bi_mode + self.input_size + self.char_vec_len,
                            self.hidden_size,
                            self.n_layers,
                            dropout=self.dropout_p,
                            batch_first=True,
                            bidirectional=True)

        self.mdn = nn.Linear(self.hidden_size * 2 * self.bi_mode, self.num_gaussian * 6 + 1)
        self.window = nn.Linear(self.hidden_size * self.bi_mode, self.num_attn_gaussian * 3)

    def read_config(self):
        """
        Read configuration file with hyperparameters.
        :return: None
        """
        config_path = os.path.join(os.getcwd(), './config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        model_params = params["CongenModelParams"]
        dataset_params = params["DatasetParams"]

        self.max_seq = dataset_params["max_seq"]

        self.input_size = model_params["input_size"]
        self.bi_dir = model_params["bi_dir"]
        self.bi_mode = model_params["bi_mode"]
        self.hidden_size = model_params["hidden_size"]
        self.n_layers = model_params["n_layers"]
        self.num_gaussian = model_params["num_gaussian"]
        self.num_attn_gaussian = model_params["num_attn_gaussian"]
        self.char_vec_len = model_params["char_vec_len"]
        self.dropout_p = model_params["dropout_p"]
        self.bias = model_params["bias"]

    def training_step(self, batch, batch_idx):
        """
        Perform train step.
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :param batch_idx: index of current batch, non applicable here
        :return: mean loss
        """
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'train')
        self.log('loss_train', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform validation step.
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :param batch_idx: index of current batch, non applicable here
        :return: mean loss
        """
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'val')
        self.log('loss_val', loss, on_step=False, on_epoch=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Perform test step.
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :param batch_idx: index of current batch, non applicable here
        :return: mean loss
        """
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        return loss

    def _shared_step(self, batch):
        mdn_params, hidden1, hidden2, loss = self.forward(batch)

        metrics = self.calc_metrics(mdn_params, batch)

        return loss, metrics

    def forward(self, batch):
        # inp, char_vec, old_k, old_w, text_len, hidden1, hidden2, bias = 0
        batch_size = batch[0][0].size(0)

        data, mask, text_len, char_to_vec, vec_to_char = batch
        text_len = text_len.unsqueeze(1)

        stroke_tensor, target_tensor, text_tensor = data
        stroke_mask, text_mask = mask

        hidden1 = self.initLHidden(batch_size)
        hidden2 = self.initLHidden(batch_size)
        old_k = torch.zeros((batch_size, self.num_attn_gaussian), dtype=torch.float)
        old_w = text_tensor.narrow(1, 0, 1)

        mdn_params = None
        loss = 0
        for stroke in range(self.max_seq):
            inp = stroke_tensor[:, stroke, :]

            if len(inp.size()) == 2:
                inp = inp.unsqueeze(1)

            embed = torch.cat([inp, old_w], dim=-1)  # adding attention window to the input of rnn

            output1, hidden1 = self.rnn1(embed, hidden1)
            if self.bi_mode == 1:
                output1 = output1[:, :, 0:self.hidden_size] + output1[:, :, self.hidden_size:]

            ##### implementing Eqn. 48 - 51 of the paper ###########
            abk_t = self.window(output1.squeeze(1)).exp()
            a_t, b_t, k_t = abk_t.split(self.num_attn_gaussian, dim=1)
            k_t = old_k + k_t
            #######################################################

            ##### implementing Eqn. 46 and 47 of the paper ###########
            u = torch.linspace(1, text_tensor.shape[1], text_tensor.shape[1])
            phi_bku = torch.exp(torch.mul(torch.sub(k_t.unsqueeze(2).repeat((1, 1, len(u))), u) ** 2,
                                          -b_t.unsqueeze(2)))
            phi = torch.sum(torch.mul(a_t.unsqueeze(2), phi_bku), dim=1) * (text_tensor.shape[1] / text_len)
            win_t = torch.sum(torch.mul(phi.unsqueeze(2), text_tensor), dim=1)
            ##########################################################

            inp_skip = torch.cat([output1, inp, win_t.unsqueeze(1)], dim=-1)  # implementing skip connection
            output2, hidden2 = self.rnn2(inp_skip, hidden2)
            if self.bi_mode == 1:
                output2 = output2[:, :, 0:self.hidden_size] + output2[:, :, self.hidden_size:]
            output = torch.cat([output1, output2], dim=-1)

            ##### implementing Eqn. 17 to 22 of the paper ###########
            y_t = self.mdn(output.squeeze(1))

            e_t = y_t[:, 0:1]
            pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t = torch.split(y_t[:, 1:], self.num_gaussian, dim=1)
            e_t = torch.sigmoid(e_t)
            pi_t = torch.softmax(pi_t * (1 + self.bias))  # bias would be used during inference
            s1_t, s2_t = torch.exp(s1_t), torch.exp(s2_t)
            rho_t = torch.tanh(rho_t)
            ##########################################################

            mdn_params = [e_t, pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t, phi, win_t, k_t]

            old_k = mdn_params[-1]
            old_w = mdn_params[-2].unsqueeze(1)
            loss += self.mdn_loss(mdn_params, target_tensor[:, stroke, :], stroke_mask[:, stroke])

        loss = loss/self.max_seq

        return mdn_params, hidden1, hidden2, loss

    #def initHidden(self):
    #    return torch.zeros(self.n_layers * self.bi, self.batch_size, self.hidden_size)

    def initLHidden(self, batch_size):
        return (torch.zeros(self.n_layers * self.bi, batch_size, self.hidden_size),
                torch.zeros(self.n_layers * self.bi, batch_size, self.hidden_size))

    def mdn_loss(self, mdn_params, data, mask=[]):

        def get_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
            ##### implementing Eqn. 24 and 25 of the paper ###########
            norm1 = torch.sub(x1.view(-1, 1), mu1)
            norm2 = torch.sub(x2.view(-1, 1), mu2)
            s1s2 = torch.mul(s1, s2)
            z = torch.div(norm1 ** 2, s1 ** 2) + torch.div(norm2 ** 2, s2 ** 2) - 2 * torch.div(
                torch.mul(rho, torch.mul(norm1, norm2)), s1s2)
            deno = 2 * np.pi * s1s2 * torch.sqrt(1 - rho ** 2)
            numer = torch.exp(torch.div(-z, 2 * (1 - rho ** 2)))
            ##########################################################
            return numer / deno

        eos, x1, x2 = data[:, 0], data[:, 1], data[:, 2]
        e_t, pi_t = mdn_params[0], mdn_params[1]
        res = get_2d_normal(x1, x2, mdn_params[2], mdn_params[3], mdn_params[4], mdn_params[5], mdn_params[6])

        epsilon = torch.tensor(1e-20, dtype=torch.float)  # to prevent overflow

        res1 = torch.sum(torch.mul(pi_t, res), dim=1)
        res1 = -torch.log(torch.max(res1, epsilon))
        res2 = torch.mul(eos, e_t.t()) + torch.mul(1 - eos, 1 - e_t.t())
        res2 = -torch.log(res2)

        if len(mask) != 0:  # using masking in case of padding
            res1 = torch.mul(res1, mask)
            res2 = torch.mul(res2, mask)
        return torch.sum(res1 + res2)

    def configure_optimizers(self):
        """
        Selection of gradient descent algorithm and learning rate scheduler.
        :return: optimizer algorithm, learning rate scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)

        return [optimizer], []

    def calc_metrics(self, prediction, target):
        """
        Calculate useful metrics.
        :param prediction: tensor of predictions (batch_size)
        :param target: tensor of ground truths (batch_size)
        :return: metric dictionary
        """
        metrics = {}

        #metrics['mae'] = torch.abs(prediction - target).mean()

        return metrics

    def log_metrics(self, metrics: dict, type: str):
        """
        Log metrics on Tensorboard.
        :param metrics: metric dictionary
        :param type: check if training or validation metrics
        :return: None
        """
        on_step = True if type == 'train' else False

        for key in metrics:
            self.log(key + '_' + type, metrics[key], on_step=on_step, on_epoch=True, logger=True)
