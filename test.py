import torch
import argparse
from utils import load_setting, load_tokenizer
from models import SwinTransformerOCR
from dataset import CustomCollate
import os
from pathlib import Path
from tqdm import tqdm

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

tokenizer = load_tokenizer(cfg.tokenizer)
model = SwinTransformerOCR(cfg, tokenizer)
saved = torch.load(cfg.checkpoint, map_location=device)
model.load_state_dict(saved['state_dict'])
collate = CustomCollate(cfg, tokenizer=tokenizer)

test_path = './dataset/test/'
test_file_info = os.walk(test_path)
images = []
for _, _, f in test_file_info:
    images = f
labels, predicts = [], []
with torch.no_grad():
    loader = tqdm(images)
    for img in loader:
        model.eval()
        x = collate.ready_image(Path(test_path + img))
        predict = model.predict(x)[0]
        predicts.append(predict)
        labels.append(img[:4])
        loader.set_description('label:{}  predict:{}'.format(img[:4], predict))
correct = 0
total = 4 * len(labels)
for i, pre in enumerate(predicts):
    for j in range(4):
        if pre[j] == labels[i][j]:
            correct += 1
print("Accuracy:{:.2f}%".format((correct / total) * 100))
