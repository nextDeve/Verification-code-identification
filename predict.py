import torch
import argparse
from utils import load_setting, load_tokenizer
from models import SwinTransformerOCR
from dataset import CustomCollate
import os
from pathlib import Path


def predict_res():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/default.yaml",
                        help="Experiment settings")
    parser.add_argument("--tokenizer", "-tk", type=str, default="checkpoints/version_0/transformer-ocr_test.pkl",
                        help="Load pre-built tokenizer")
    parser.add_argument("--checkpoint", "-c", type=str,
                        default="checkpoints/version_0/checkpoints-epoch=31-accuracy=0.98267.ckpt",
                        help="Load model weight in checkpoint")
    args = parser.parse_args()

    cfg = load_setting(args.setting)
    cfg.update(vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load
    tokenizer = load_tokenizer(cfg.tokenizer)
    model = SwinTransformerOCR(cfg, tokenizer)
    saved = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(saved['state_dict'])
    collate = CustomCollate(cfg, tokenizer=tokenizer)
    model.eval()
    target = './extradata/flask/save/'
    for _, _, f in os.walk(target):
        with torch.no_grad():
            x = collate.ready_image(Path(target + f[0]))
            return model.predict(x)[0]
