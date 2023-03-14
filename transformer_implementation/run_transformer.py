from data_load import *
from model_transformer import TransformerModel, ModelUncond, mdn_loss_transformer, sample_uncond, scheduled_sample
from eval_hand import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transformer parameters
n_heads = 8
hidden_size = 400
n_layers = 2

num_gaussian = 20
dropout = 0.2
batch_size = 32
max_seq = 400
print_every = batch_size*40
plot_every = 4

learning_rate = 0.0005    
print_loss = 0
total_loss = torch.Tensor([0]).to(device)
print_loss_total = 0 
teacher_forcing_ratio = 1   # can be used to switch to scheduled sampling; do not change it now as scheduled sampling is unstable
clip = 10.0 
epochs = 21

data_x, data_y = get_data_uncond(batch_size=6000, max_seq = max_seq)

lr_model = ModelUncond(input_size=3, n_heads=n_heads, hidden_size=hidden_size,
                       n_layers=n_layers, num_gaussian=num_gaussian, dropout=dropout).to(device)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(count_parameters(lr_model))

model_optimizer = optim.Adam(lr_model.parameters(), lr=learning_rate)

src_mask = TransformerModel.generate_square_subsequent_mask(max_seq+1).to(device)

num_mini_batch = len(data_x) - (len(data_x) % batch_size)

for big_epoch in range(epochs):
    start = time.time()
    print_loss_total = 0
    for i in range(0, num_mini_batch, batch_size):
      input_tensor = torch.tensor(data_x[i:i+batch_size], dtype=torch.float, device=device)
      target_tensor = torch.tensor(data_y[i:i+batch_size], dtype=torch.float, device=device)

      # NOT OPTIMAL - TODO: make sequence-wise form below work
      loss = 0
      mdn_params = lr_model(input_tensor, src_mask)
      for stroke in range(0,input_tensor.size()[1]):
        out_sample = target_tensor[:,stroke,:]
        loss += mdn_loss_transformer(stroke, mdn_params, out_sample)
      loss = loss/input_tensor.size()[1]

      # mdn_params = lr_model(input_tensor, src_mask)
      # out_sample = target_tensor[:,:input_tensor.size()[1],:]
      # loss = mdn_loss_transformer(mdn_params, out_sample)/input_tensor.size()[1]
      
      loss.backward()         
      torch.nn.utils.clip_grad_norm(lr_model.parameters(), clip)    
      model_optimizer.step()
      
      print_loss_total += loss.item()/target_tensor.size()[1]
      
      if i % print_every == 0 and i>0:
          print_loss_avg = print_loss_total / print_every
          print_loss_total = 0
          print('%d  %s (%d %d%%) %.4f' % (big_epoch,timeSince(start, i / num_mini_batch),
                                              i, i / num_mini_batch * 100, print_loss_avg))
      print_loss+=1
          
    if big_epoch % plot_every == 0 and big_epoch>0:
        a,b = sample_uncond(lr_model,time_step=800)
        plot_stroke(a)
            
save_checkpoint(big_epoch, lr_model, model_optimizer, 'saved_model', \
                    filename='model_uncond.pt')