import torch
import pytorch_lightning as pl
import argparse
from pathlib import Path
import os
from models import SwinTransformerOCR
from dataset import CustomDataset, CustomCollate, Tokenizer
from utils import load_setting, save_tokenizer, CustomTensorBoardLogger, load_tokenizer

from torch.utils.data import DataLoader

if __name__ == "__main__":
    # 添加参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/default.yaml",
                        help="Experiment settings")
    parser.add_argument("--version", "-v", type=int, default=0,
                        help="Train experiment version")
    parser.add_argument("--load_tokenizer", "-bt", type=str, default="",
                        help="Load pre-built tokenizer")
    parser.add_argument("--num_workers", "-nw", type=int, default=0,
                        help="Number of workers for dataloader")
    parser.add_argument("--batch_size", "-bs", type=int, default=4,
                        help="Batch size for training and validate")
    parser.add_argument("--resume_train", "-rt", type=str, default="",
                        help="Resume train from certain checkpoint")
    args = parser.parse_args()

    # 加载参数
    cfg = load_setting(args.setting)
    cfg.update(vars(args))

    # ----- dataset -----
    # 构建数据集，也就是模型能识别的数据输入格式
    train_set = CustomDataset(cfg, cfg.train_data)
    val_set = CustomDataset(cfg, cfg.val_data)

    # ----- tokenizer -----
    # 这里就是加载已有的字符字典 或者创建字符字典
    if cfg.load_tokenizer:
        tokenizer = load_tokenizer(cfg.load_tokenizer)
    else:
        tokenizer = Tokenizer(train_set.token_id_dict)
        os.makedirs(Path(cfg.save_path) / f"version_{cfg.version}", exist_ok=True)
        save_path = "{}/{}/{}.pkl".format(cfg.save_path, f"version_{cfg.version}", cfg.name.replace(' ', '_'))
        save_tokenizer(tokenizer, save_path)

    # 定义图片，验证码图片，转换函数
    train_collate = CustomCollate(cfg, tokenizer, is_train=True)
    # 定义图片，验证码图片，转换函数
    val_collate = CustomCollate(cfg, tokenizer, is_train=False)

    # 将数据加载到DataLoader中，并使用定义好的转换函数，转换图片为特定格式
    train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers, collate_fn=train_collate)
    valid_dataloader = DataLoader(val_set, batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers, collate_fn=val_collate)

    cfg.num_train_step = len(train_dataloader)
    # 定义模型
    model = SwinTransformerOCR(cfg, tokenizer)
    # 定义日志
    logger = CustomTensorBoardLogger("tb_logs", name="model", version=cfg.version,
                                     default_hp_metric=False)
    # 定义模型检查点，以验证集正确率为衡量标准，保存再验证集中识别正确率最高的3个模型
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="accuracy",
        dirpath=f"{cfg.save_path}/version_{cfg.version}",
        filename="checkpoints-{epoch:02d}-{accuracy:.5f}",
        save_top_k=3,
        mode="max",
    )
    # 定义学习率检查点
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # 获取可用模型训练设备
    device_cnt = torch.cuda.device_count()

    # 设备使用策略 有几张显卡就用几张显卡，没有就用cpu
    strategy = pl.plugins.DDPPlugin(find_unused_parameters=False) if device_cnt > 1 else None
    # 定义训练器
    trainer = pl.Trainer(gpus=device_cnt,
                         max_epochs=cfg.epochs,
                         logger=logger,
                         num_sanity_val_steps=1,
                         strategy=strategy,
                         callbacks=[ckpt_callback, lr_callback],
                         resume_from_checkpoint=cfg.resume_train if cfg.resume_train else None)
    # 模型训练
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
