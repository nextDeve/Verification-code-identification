####  如果需要自己训练模型

1.重新生成数据集，与原来生成方式一样，运行captcha.py

2.预处理，运行preprocess.py,将把生成的数据集分为训练集、验证集、测试集，保存在dataset文件夹下

3.运行run.py，如果不想覆盖之前的训练记录，修改run.py中的版本号即可，如下：

```python
# 修改  default=自己的版本号即可
parser.add_argument("--version", "-v", type=int, default=0,help="Train experiment version")
```

训练完成后，将会得到新的模型，存放在checkpoint文件夹下，需要使用自己训练的模型，需要修改两个地方:

```python
# 修改tokenizer的default值，改为你的版本对应的checkpoint文件夹下的transformer-ocr_test.pkl
parser.add_argument("--tokenizer", "-tk", type=str, default="checkpoints/version_0/transformer-ocr_test.pkl",help="Load pre-built tokenizer")
# 修改模型权重路径  修改为改为你的版本对应的checkpoint文件夹下的模型权重，保存了3表现最好的模型，都可以用
parser.add_argument("--checkpoint", "-c", type=str,default="checkpoints/version_0/checkpoints-epoch=31-accuracy=0.98267.ckpt",help="Load model weight in checkpoint")
```

