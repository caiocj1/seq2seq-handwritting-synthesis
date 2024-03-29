import torch
import numpy as np


def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def get_pi_id(x, dist):
    # implementing the cumulative index retrieval
    N = dist.shape[0]
    accumulate = 0
    for i in range(0, N):
        accumulate += dist[i]
        if (accumulate >= x):
            return i
    return -1

def sample_congen(lr_model,
                  text,
                  char_to_vec,
                  hidden_size,
                  start=[0, 0, 0],
                  time_step=3000,
                  scale=50,
                  bias1=1,
                  bias2=1,
                  num_attn_gaussian=10,
                  bi_dir=True,
                  random_state=56):

    np.random.seed(random_state)

    if bi_dir == True:
        bi = 2
    else:
        bi = 1

    prev_x = torch.tensor(start, dtype=torch.float)
    prev_x[0] = 1
    strokes = np.zeros((time_step, 3), dtype=np.float32)
    old_k = torch.zeros((1, num_attn_gaussian), dtype=torch.float)
    text_len = torch.tensor([[len(text)]], dtype=torch.float)

    vectors = np.zeros((len(text), len(char_to_vec) + 1))

    for p, q in enumerate(text):
        try:
            vectors[p][char_to_vec[q]] = 1
        except:
            vectors[p][-1] = 1
            continue

    text_tensor = torch.tensor(vectors, dtype=torch.float)
    old_w = text_tensor.narrow(0, 0, 1).unsqueeze(0)

    phis, win = [], []
    count = 0
    stop = False

    mixture_params = []
    hidden1 = (torch.zeros(bi, 1, hidden_size), torch.zeros(bi, 1, hidden_size))
    hidden2 = (torch.zeros(bi, 1, hidden_size), torch.zeros(bi, 1, hidden_size))


    for i in range(time_step):
        mdn_params, hidden1, hidden2 = lr_model.sample(prev_x.unsqueeze(0),
                                                       text_tensor.unsqueeze(0),
                                                       old_k,
                                                       old_w,
                                                       text_len,
                                                       hidden1,
                                                       hidden2,
                                                       bias1)
        old_k = mdn_params[-1]
        old_w = mdn_params[-2].unsqueeze(1)
        idx = get_pi_id(np.random.random(), mdn_params[1][0])
        eos = 1 if np.random.random() < mdn_params[0][0] else 0

        log_s1 = mdn_params[4][0][idx].log() - bias2
        log_s2 = mdn_params[5][0][idx].log() - bias2
        log_s1 = log_s1.exp()
        log_s2 = log_s2.exp()

        next_x1, next_x2 = sample_gaussian_2d(mdn_params[2][0][idx].detach().cpu().numpy(),
                                              mdn_params[3][0][idx].detach().cpu().numpy(),
                                              log_s1.detach().cpu().numpy(),
                                              log_s2.detach().cpu().numpy(),
                                              mdn_params[6][0][idx].detach().cpu().numpy())

        strokes[i, :] = [eos, next_x1, next_x2]
        mixture_params.append([float(mdn_params[2][0][idx].detach().cpu()),
                               float(mdn_params[3][0][idx].detach().cpu()),
                               float(log_s1.detach().cpu()), float(log_s2.detach().cpu()),
                               float(mdn_params[6][0][idx].detach().cpu()), eos])
        # prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        prev_x[0], prev_x[1], prev_x[2] = eos, next_x1, next_x2

        phis.append(mdn_params[-3].squeeze(0))
        win.append(mdn_params[-2])
        old_phi = mdn_params[-3].squeeze(0)
        old_phi = old_phi.data.cpu().numpy()

        if count >= 40 and np.max(old_phi) == old_phi[-1]:
            stop = True
        else:
            count += 1

    phis = torch.stack(phis).data.cpu().numpy().T
    win = torch.stack(win).data.cpu().numpy().T
    mix_params = np.array(mixture_params)
    mix_params[:, :2] = np.cumsum(mix_params[:, :2], axis=0)

    # phi_window_plots(phis,win.squeeze(1))
    # gauss_params_plot()
    strokes[:, 1:3] *= scale  # scaling the output strokes
    return strokes[:count + scale, :],\
           mix_params[:count + scale, :], \
           phis[:, :count + scale],\
           win.squeeze(1)[:, :count + scale]


def sample_uncond(lr_model, hidden_size, start=[0, 0, 0], rnn_type=2, \
                  time_step=600, scale=20, bi_dir=True, random_state=98):
    np.random.seed(random_state)

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

    if bi_dir == True:
        bi = 2
    else:
        bi = 1

    prev_x = torch.tensor(start, dtype=torch.float)
    prev_x[0] = 1
    strokes = np.zeros((time_step, 3), dtype=np.float32)
    mixture_params = []
    if rnn_type == 1:
        hidden1 = torch.zeros(bi, 1, hidden_size)
        hidden2 = torch.zeros(bi, 1, hidden_size)
    else:
        hidden1 = (torch.zeros(bi, 1, hidden_size), torch.zeros(bi, 1, hidden_size))
        hidden2 = (torch.zeros(bi, 1, hidden_size), torch.zeros(bi, 1, hidden_size))

    for i in range(time_step):
        mdn_params, hidden1, hidden2 = lr_model.sample(prev_x.unsqueeze(0), hidden1, hidden2)
        idx = get_pi_id(np.random.random(), mdn_params[1][0])
        eos = 1 if np.random.random() < mdn_params[0][0] else 0

        next_x1, next_x2 = sample_gaussian_2d(mdn_params[2][0][idx].detach().cpu().numpy(),
                                              mdn_params[3][0][idx].detach().cpu().numpy(),
                                              mdn_params[4][0][idx].detach().cpu().numpy(),
                                              mdn_params[5][0][idx].detach().cpu().numpy(),
                                              mdn_params[6][0][idx].detach().cpu().numpy())

        mixture_params.append([float(mdn_params[2][0][idx].detach().cpu()),
                               float(mdn_params[3][0][idx].detach().cpu()),
                               float(mdn_params[4][0][idx].detach().cpu()),
                               float(mdn_params[5][0][idx].detach().cpu()),
                               float(mdn_params[6][0][idx].detach().cpu()), eos])

        strokes[i, :] = [eos, next_x1, next_x2]
        # prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        prev_x[0], prev_x[1], prev_x[2] = eos, next_x1, next_x2

    strokes[:, 1:3] *= scale
    mix_params = np.array(mixture_params)
    mix_params[:, :2] = np.cumsum(mix_params[:, :2], axis=0)
    return strokes, mix_params

def sample_uncond_attn(lr_model, hidden_size, start=[0, 0, 0], rnn_type=2, \
                  time_step=600, scale=20, bi_dir=True, random_state=98):
    np.random.seed(random_state)

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

    prev_x = torch.tensor(start, dtype=torch.float)
    prev_x[0] = 1
    strokes = np.zeros((time_step, 3), dtype=np.float32)
    mixture_params = []

    if rnn_type == 1:
        hidden1 = torch.zeros(1, 1, hidden_size)
        hidden2 = torch.zeros(1, 1, hidden_size)
    else:
        hidden1 = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
        hidden2 = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    hidden1_list = [hidden1[0][-1][None]]
    for i in range(time_step):
        mdn_params, hidden1, hidden2, attn_weights = lr_model.sample(prev_x.unsqueeze(0), hidden1, hidden2, hidden1_list)
        hidden1_list.append(hidden1[0][-1][None])
        if len(hidden1_list) > lr_model.att_len:
            hidden1_list.pop(0)
        idx = get_pi_id(np.random.random(), mdn_params[1][0])
        eos = 1 if np.random.random() < mdn_params[0][0] else 0

        next_x1, next_x2 = sample_gaussian_2d(mdn_params[2][0][idx].detach().cpu().numpy(),
                                              mdn_params[3][0][idx].detach().cpu().numpy(),
                                              mdn_params[4][0][idx].detach().cpu().numpy(),
                                              mdn_params[5][0][idx].detach().cpu().numpy(),
                                              mdn_params[6][0][idx].detach().cpu().numpy())

        mixture_params.append([float(mdn_params[2][0][idx].detach().cpu()),
                               float(mdn_params[3][0][idx].detach().cpu()),
                               float(mdn_params[4][0][idx].detach().cpu()),
                               float(mdn_params[5][0][idx].detach().cpu()),
                               float(mdn_params[6][0][idx].detach().cpu()), eos])

        strokes[i, :] = [eos, next_x1, next_x2]
        prev_x[0], prev_x[1], prev_x[2] = eos, next_x1, next_x2

    strokes[:, 1:3] *= scale
    mix_params = np.array(mixture_params)
    mix_params[:, :2] = np.cumsum(mix_params[:, :2], axis=0)
    return strokes, mix_params, attn_weights

def sample_prime(lr_model, text, chosen_writing_idx, char_to_vec, hidden_size, time_step=2000, scale = 50, rnn_type = 2, bias1 = 0.2, bias2 = 0.2, num_attn_gaussian = 10, bi_dir=True, random_state= 56):
    np.random.seed(random_state)

    strokes = np.load('data/strokes.npy', encoding='latin1', allow_pickle=True)
    with open('data/sentences.txt', 'r') as f:
        texts = f.readlines()


    texts = [a.split('\n')[0] for a in texts]
    start_stroke = strokes[chosen_writing_idx]
    start_text = texts[chosen_writing_idx]
    
    
    if bi_dir == True:
        bi = 2
    else:
        bi = 1

    text_concat = start_text + ' ' + text
    prev_x = torch.tensor(start_stroke, dtype=torch.float)[0]

    strokes = np.zeros((time_step, 3), dtype=np.float32)
    old_k = torch.zeros((1,num_attn_gaussian), dtype=torch.float)

    text_len = torch.tensor([[len(text)]], dtype=torch.float)
    
    vectors = np.zeros((len(text_concat), len(char_to_vec)+1))

    for p,q in enumerate(text_concat):
        try:
            vectors[p][char_to_vec[q]] = 1
        except:
            vectors[p][-1] = 1
            continue

        
    text_tensor = torch.tensor(vectors, dtype=torch.float)
    old_w = text_tensor.narrow(0,0,1).unsqueeze(0)

    phis, win = [],[]
    count = 0
    count_after_max = 0
    flag = False

    mixture_params = []
    if rnn_type == 1:
        hidden1 =  torch.zeros(bi, 1, hidden_size)
        hidden2 =  torch.zeros(bi, 1, hidden_size)
    else:
        hidden1 = (torch.zeros(bi, 1, hidden_size), torch.zeros(bi, 1, hidden_size))
        hidden2 = (torch.zeros(bi, 1, hidden_size), torch.zeros(bi, 1, hidden_size))
    
    for i in range(time_step):
        mdn_params, hidden1,hidden2 = lr_model.sample(prev_x.unsqueeze(0),text_tensor.unsqueeze(0),
                                               old_k, old_w, text_len, hidden1, hidden2, bias1)
        old_k = mdn_params[-1]
        old_w = mdn_params[-2].unsqueeze(1)
        idx = get_pi_id(np.random.random(), mdn_params[1][0])
        eos = 1 if np.random.random() < mdn_params[0][0] else 0
        
        log_s1 = mdn_params[4][0][idx].log() - bias2
        log_s2 = mdn_params[5][0][idx].log() - bias2
        log_s1 = log_s1.exp()
        log_s2 = log_s2.exp()

        next_x1, next_x2 = sample_gaussian_2d(mdn_params[2][0][idx].detach().cpu().numpy(), mdn_params[3][0][idx].detach().cpu().numpy(), 
                            log_s1.detach().cpu().numpy(), log_s2.detach().cpu().numpy(), 
                            mdn_params[6][0][idx].detach().cpu().numpy())
        
        mixture_params.append([float(mdn_params[2][0][idx].detach().cpu()), 
                               float(mdn_params[3][0][idx].detach().cpu()), 
                            float(log_s1.detach().cpu()), float(log_s2.detach().cpu()), 
                            float(mdn_params[6][0][idx].detach().cpu()), eos])

        if i < len(start_stroke):
            prev_x = torch.tensor(start_stroke, dtype=torch.float)[i]
            zero_timestep = i
            #eos0, next_x10, next_x20 = eos, next_x1, next_x2
        else:
            # prev_x = np.zeros((1, 1, 3), dtype=np.float32)
            prev_x[0],prev_x[1],prev_x[2] = eos, next_x1, next_x2
            strokes[i-zero_timestep, :] = [eos, next_x1, next_x2]

            phis.append(mdn_params[-3].squeeze(0))
            win.append(mdn_params[-2])
            old_phi = mdn_params[-3].squeeze(0)
            old_phi = old_phi.data.cpu().numpy()     

            count+=1

            if old_phi[-1]/np.sum(old_phi) > 0.9:
                flag = True

            if flag:
                count_after_max+=1

            if old_phi[-1]/np.sum(old_phi) > 0.99 or count_after_max>=150:
                break
    
    unstacked_phis = phis
    phis = torch.stack(phis).data.cpu().numpy().T
    win = torch.stack(win).data.cpu().numpy().T
    mix_params = np.array(mixture_params)
    mix_params[:,:2] = np.cumsum(mix_params[:,:2], axis=0)
    
    strokes[:, 1:3] *= scale  # scaling the output strokes
    
    return strokes[1:count+scale,:], mix_params[:count+scale,:],\
             phis[:,:count+scale], win.squeeze(1)[:,:count+scale], start_stroke, unstacked_phis