import copy

import torch.optim
from pytorch_lightning import LightningModule
import torch.nn as nn
import os
import yaml
from yaml import SafeLoader
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
import torchvision

from utils.sampling import sample_uncond_attn
from utils.plots import plot_stroke, plot_stroke_attentions

# CHANGING HIDDEN LIST

class seq2seqAtt(nn.Module):
    '''
    concat global attention a la Luong et al. 2015 (subsection 3.1): https://arxiv.org/pdf/1508.04025.pdf
    here we use the same ideia of their cross-attention mechanism, but for self-attetion with the target being the last hidden state of the RNN
    '''
    def __init__(self, hidden_dim, hidden_dim_s, hidden_dim_t):
        super(seq2seqAtt, self).__init__()
        self.ff_concat = nn.Linear(hidden_dim_s+hidden_dim_t, hidden_dim)
        self.ff_score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, target_h, source_hs):
        """
        target_h: (batch, 1, feat)
        source_hs: (batch, seq, feat)
        implement the score computation part of the concat formulation (see section 3.1. of Luong 2015)

        returns contexts (batch, 1, feat) and attention scores (batch, seq)
        """
        target_h_rep = target_h.repeat(1, source_hs.size(1), 1) # (batch, 1, feat) -> (batch, seq, feat)
        concat_output = torch.concat([source_hs, target_h_rep], axis=2) # (batch, seq, 2*feat)
        scores = self.ff_score(self.ff_concat(concat_output)) # (batch, seq, 1)
        scores = scores.squeeze(dim=2) # (batch, seq, 1) -> (batch, seq)
        norm_scores = torch.softmax(scores, 1) # (batch, seq)
        source_hs_p = source_hs.permute((2, 0, 1)) # (batch, seq, feat) -> (feat, batch, seq)
        weighted_source_hs = (norm_scores * source_hs_p) # (batch, seq) * (feat, batch, seq) (* checks from right to left that the dimensions match)
        ct = torch.sum(weighted_source_hs.permute((1, 2, 0)), 1, keepdim=True) # (feat, batch, seq) -> (batch, seq, feat) -> (batch, 1, feat); keepdim otherwise sum squeezes
        return ct, norm_scores

class UncondModelAttention(LightningModule):

    def __init__(self):
        super(UncondModelAttention, self).__init__()

        # Read config file if needed
        self.read_config()

        self.rnn1 = nn.LSTM(self.input_size,
                            self.hidden_size,
                            self.n_layers,
                            dropout=self.dropout_p,
                            batch_first=True)

        self.rnn2 = nn.LSTM(self.hidden_size + self.input_size,
                            self.hidden_size,
                            self.n_layers,
                            dropout=self.dropout_p,
                            batch_first=True)

        self.att_mech = seq2seqAtt(self.hidden_size, self.hidden_size, self.hidden_size)

        self.ff_concat = nn.Linear(2*self.hidden_size, self.hidden_size)

        self.mdn = nn.Linear(self.hidden_size, self.num_gaussian * 6 + 1)

    def read_config(self):
        """
        Read configuration file with hyperparameters.
        :return: None
        """
        config_path = os.path.join(os.getcwd(), './config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        model_params = params["UncondModelAttentionParams"]
        dataset_params = params["DatasetParams"]

        self.max_seq = dataset_params["max_seq_uncond"]

        self.input_size = model_params["input_size"]
        # self.bi_dir = model_params["bi_dir"]
        # self.bi_mode = model_params["bi_mode"]
        self.hidden_size = model_params["hidden_size"]
        self.n_layers = model_params["n_layers"]
        self.num_gaussian = model_params["num_gaussian"]
        self.dropout_p = model_params["dropout_p"]
        self.bias = model_params["bias"]
        self.att_len = model_params["att_len"]

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
        batch_size = batch[0].size(0)

        input_tensor = batch[0]
        target_tensor = batch[1]

        device = input_tensor.device
        hidden1 = self.initLHidden(batch_size, device)
        hidden2 = self.initLHidden(batch_size, device)

        mdn_params = None
        hidden1_list = self.att_len * [hidden1[0][-1][None]] # list of hidden sizes
        loss = 0
        for stroke in range(self.max_seq):
            mdn_params, hidden1, hidden2, attn_weights = self.sample(input_tensor[:, stroke, :],
                                                       hidden1,
                                                       hidden2,
                                                       hidden1_list)
            hidden1_list.append(hidden1[0][-1][None])
            if len(hidden1_list) > self.att_len:
                hidden1_list.pop(0)
            out_sample = target_tensor[:, stroke, :]

            loss += self.mdn_loss(mdn_params, out_sample)
        loss = loss/self.max_seq

        return mdn_params, hidden1, hidden2, loss

    # sample using attention
    def sample(self, inp, hidden1, hidden2, hidden1_list):
        if len(inp.size()) == 2:
            embed = inp.unsqueeze(1)
        else:
            embed = inp

        # encoder
        output1, hidden1 = self.rnn1(embed.float(), hidden1)
        
        # sequence of final hidden states
        hidden_seq1 = torch.cat(hidden1_list, dim=0).permute((1, 0, 2)) # (batch, seq, feat)

        # implementing attention
        source_context = self.att_mech(output1, hidden_seq1)
        attn_weights = source_context[1]    # useful for visualization
        source_context =  source_context[0] # (batch, 1, feat)

        # tilde_h1 = tanh(W_c[c_t;h_t]) - Luong et Al
        tilde_h1 = nn.Tanh()(self.ff_concat(torch.cat([source_context, output1], axis=-1))) # (batch, 1, feat)

        inp_skip = torch.cat([tilde_h1, embed], dim=-1)  # implementing skip connection
        
        # decoder
        output2, hidden2 = self.rnn2(inp_skip.float(), hidden2)

        ##### implementing Eqn. 17 to 22 of the paper ###########
        y_t = self.mdn(output2.squeeze(1)) # (batch, feat)
        e_t = y_t[:, 0:1]

        pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t = torch.split(y_t[:, 1:], self.num_gaussian, dim=1)
        e_t = torch.sigmoid(e_t)
        pi_t = torch.softmax(pi_t, dim=-1)
        s1_t, s2_t = torch.exp(s1_t), torch.exp(s2_t)
        rho_t = torch.tanh(rho_t)
        #######################################################

        mdn_params = [e_t, pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t]
        return mdn_params, hidden1, hidden2, attn_weights

    def initLHidden(self, batch_size, device):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size).float().to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size).float().to(device))

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

    def on_train_epoch_end(self):
        char_list = ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,."\'?-!'
        char_to_code = {}
        c = 0
        for ch in char_list:
            char_to_code[ch] = c
            c += 1

        with torch.no_grad():
            model = copy.deepcopy(self).to("cpu")
            strokes, mix_params, attn_weights = sample_uncond_attn(model, self.hidden_size)
            print("final attention weights:", attn_weights)
            fig = plot_stroke_attentions(strokes, attn_weights[0], return_fig=True)
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            img_tensor = torchvision.transforms.ToTensor()(img)
            self.logger.experiment.add_image(f'uncond_viz/sample', img_tensor, self.global_step)
