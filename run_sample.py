from utils.loading import load_pretrained_congen, load_pretrained_uncond
from utils.plots import plot_stroke
from utils.sampling import sample_congen, sample_uncond

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", choices=["cond", "uncond"], required=True)
    parser.add_argument("--ckpt_path", "-w", default=None)
    parser.add_argument('--text', '-t', default=None)
    parser.add_argument("--index", "-i", help="Index of dataset to base calligraphy on, for primed sampling.", default=None)
    args = parser.parse_args()

    if args.model == "cond":
        assert args.text is not None, "Must give text when using conditional model."

        lr_model, char_to_vec, h_size = load_pretrained_congen(args.ckpt_path)
        strokes, mix_params, phi, win = sample_congen(lr_model, args.text, char_to_vec, h_size)

        plot_stroke(strokes)
    elif args.model == "uncond":
        lr_model, h_size = load_pretrained_uncond(args.ckpt_path)
        strokes, mix_params = sample_uncond(lr_model, h_size)

        plot_stroke(strokes)