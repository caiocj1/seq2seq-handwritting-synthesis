# Seq2Seq Handwriting Synthesis

Implementation of "Generating Sequences with Recurrent Neural Networks" (https://arxiv.org/abs/1308.0850).

Reutilized code from https://github.com/dataflowr/Project-DL-Seq2Seq/tree/master/handwriting%20synthesis, but created 
new repository to refactor into PyTorch Lightning.

Contributions to original repo:
- Added primed sampling.
- Added model with attention mechanism on RNN.
- Added transformer model.
- Added visualizations during training.

***

### Usage:
1. To launch training, ``python run_training -v <run_name> -m <model_name>``.

Available models: ``["cond", "uncond", "attn_cond", "attn_uncond"]``. Default: `"cond"`.

2. To track training, ``tensorboard --logdir lightning_logs --bind_all``.

Images generated at the end of each epoch are available in TensorBoard.

3. To load weights and see sample generation, ``python run_sample -m <model_name> -w <path/to/ckpt.ckpt>``.

Optional arguments: ``-i <index_of_training_set> -t <text_to_generate>``. These arguments are used for primed 
sampling and for the 
conditional model, respectively.

***