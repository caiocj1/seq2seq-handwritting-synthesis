import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-np.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
    
class TransformerModel(nn.Module):
    def __init__(self, input_size, n_heads, hidden_size, n_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        '''
        input_size: number of inpit features
        hidden_size: the hidden dimension of the model.
        We assume that embedding_dim = hidden_size
        n_layers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        n_heads: the number of heads in the multiheadattention models
        dropout: the dropout value
         '''
        self.model_type = "Transformer"
        self.encoder = nn.Linear(input_size,hidden_size) # should we keep it?
        self.pos_encoder = PositionalEncoding(hidden_size,dropout)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size,n_heads,hidden_size,dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,n_layers)
        self.hidden_size = hidden_size
        self.init_weights()

    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * np.sqrt(self.hidden_size) 
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output

class MixtureWeights(nn.Module):
    def __init__(self, hidden_size, num_gaussian):
        super(MixtureWeights, self).__init__()
        self.mdn = nn.Linear(hidden_size, num_gaussian*6+1)
        self.num_gaussian = num_gaussian
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.mdn.bias.data.zero_()
        self.mdn.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        ##### implementing Eqn. 17 to 22 of the paper ###########
        y_t = self.mdn(src)
        e_t = y_t[:,:,0:1]
        
        pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t = torch.split(y_t[:,:,1:], self.num_gaussian, dim=-1)
        e_t = F.sigmoid(e_t)
        pi_t = F.softmax(pi_t)
        s1_t, s2_t = torch.exp(s1_t), torch.exp(s2_t)
        rho_t = torch.tanh(rho_t)
        #######################################################
        
        mdn_params = [e_t, pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t]
        return mdn_params

class ModelUncond(nn.Module):
    def __init__(self, input_size, n_heads, hidden_size, n_layers=2, num_gaussian=2, dropout=0.5):
        super(ModelUncond, self).__init__()
        self.base = TransformerModel(input_size, n_heads, hidden_size, n_layers, dropout)
        self.decoder = MixtureWeights(hidden_size, num_gaussian)
        
    def forward(self, src, src_mask):
        # base model
        x = self.base(src, src_mask)
        # MDN parameters
        mdn_params = self.decoder(x)
        return mdn_params
    
class ModelCond(nn.Module):
    pass

# NOT OPTIMAL - TODO: make sequence-wise form work
def mdn_loss_transformer(stroke, mdn_params, data, mask=[]):

    def get_2d_normal(x1,x2,mu1,mu2,s1,s2,rho):
        
      ##### implementing Eqn. 24 and 25 of the paper ###########
      norm1 = torch.sub(x1.view(-1,1),mu1)
      norm2 = torch.sub(x2.view(-1,1),mu2)
      s1s2 = torch.mul(s1,s2)
      z = torch.div(norm1**2,s1**2) + torch.div(norm2**2,s2**2) - 2*torch.div(torch.mul(rho, torch.mul(norm1,norm2)),s1s2)
      deno = 2*np.pi*s1s2*torch.sqrt(1-rho**2)
      numer = torch.exp(torch.div(-z,2*(1-rho**2)))
      ##########################################################
      return numer / deno

    eos, x1, x2 = data[:,0], data[:,1], data[:,2]
    e_t, pi_t = mdn_params[0][:,stroke,:], mdn_params[1][:,stroke,:]
    res = get_2d_normal(x1,x2,mdn_params[2][:,stroke,:],mdn_params[3][:,stroke,:],
                        mdn_params[4][:,stroke,:],mdn_params[5][:,stroke,:],mdn_params[6][:,stroke,:])
    
    epsilon = torch.tensor(1e-20, dtype=torch.float, device=device)  # to prevent overflow

    res1 = torch.sum(torch.mul(pi_t,res),dim=1)
    res1 = -torch.log(torch.max(res1,epsilon))
    res2 = torch.mul(eos, e_t.t()) + torch.mul(1-eos,1-e_t.t())
    res2 = -torch.log(res2)
    
    if len(mask)!=0:        # using masking in case of padding
        res1 = torch.mul(res1,mask)
        res2 = torch.mul(res2,mask)
    return torch.sum(res1+res2)

def get_pi_id(x, dist):    
    # implementing the cumulative index retrieval
    N = dist.shape[0]
    accumulate = 0
    for i in range(0, N):
        accumulate += dist[i]
        if (accumulate >= x):
            return i
    return -1

def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def scheduled_sample(lr_model, hidden_size, prev_x, batch_size = 50,  bi_dir=True, rnn_type=2, random_state=123):
    pass

def sample_uncond(model, start=[0,0,0], time_step=600, scale = 20, random_state= 98):
    np.random.seed(random_state)
    prev_x = torch.tensor(start,dtype=torch.float, device=device)
    prev_x[0] = 1
    prev_x = prev_x.unsqueeze(0)
    strokes = np.zeros((time_step, 3), dtype=np.float32)
    mixture_params = []

    for i in range(time_step):
        src_mask = TransformerModel.generate_square_subsequent_mask(i+1).to(device)
        mdn_params = model(prev_x.unsqueeze(0).to(device),src_mask=src_mask)
        idx = get_pi_id(np.random.random(), mdn_params[1][0][-1])    # taking last element from sequence
        eos = 1 if np.random.random() < mdn_params[0][0][-1] else 0

        next_x1, next_x2 = sample_gaussian_2d(mdn_params[2][0][-1][idx].detach().cpu().numpy(), mdn_params[3][0][-1][idx].detach().cpu().numpy(), 
                            mdn_params[4][0][-1][idx].detach().cpu().numpy(), mdn_params[5][0][-1][idx].detach().cpu().numpy(), 
                            mdn_params[6][0][-1][idx].detach().cpu().numpy())
        
        mixture_params.append([float(mdn_params[2][0][-1][idx].detach().cpu()), 
                               float(mdn_params[3][0][-1][idx].detach().cpu()), 
                            float(mdn_params[4][0][-1][idx].detach().cpu()), 
                            float(mdn_params[5][0][-1][idx].detach().cpu()), 
                            float(mdn_params[6][0][-1][idx].detach().cpu()), eos])
        
        strokes[i, :] = [eos, next_x1, next_x2]
        prev_x = torch.cat([prev_x, torch.tensor([eos, next_x1, next_x2], dtype=torch.float).unsqueeze(0).to(device)], dim=0)
        
    strokes[:, 1:3] *= scale
    mix_params = np.array(mixture_params)
    mix_params[:,:2] = np.cumsum(mix_params[:,:2], axis=0)
    return strokes, mix_params

def sample_congen(model, text, char_to_vec, hidden_size, start=[0,0,0], time_step=1000, scale = 50,\
                rnn_type = 2, bias1 = 1, bias2 = 1, num_attn_gaussian = 10, bi_dir=True, random_state= 56):
    np.random.seed(random_state)
    pass

def sample_prime(model, text, start_text, start_stroke, char_to_vec, hidden_size, time_step=2000, scale = 50,\
                rnn_type = 2, bias1 = 1, bias2 = 1, num_attn_gaussian = 10, bi_dir=True, random_state= 56):
    np.random.seed(random_state)
    pass