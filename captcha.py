# coding=utf-8

import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
from tqdm import tqdm

# 生成验证码存储路径
filename = "./dataset/images/"
os.makedirs(filename, exist_ok=True)
# 系统字体路径
font_path = './Fonts/georgia.ttf'
# 生成验证码的位数
number = 4
# 生成验证码图片的宽度和高度
size = (150, 55)
# 背景颜色，默认设为白色
bgcolor = (255, 255, 255)
# 是否要加入干扰线
draw_line = True
# 图像中噪点数目
nois = 30


# 随机选取一个字符
def gene_text():
    # source = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I','J', 'K','L', 'M', 'N','O','P','Q','R','S', 'T', 'U', 'V', 'W', 'Z','X', 'Y','0','1','2','3','4','5','6','7','8','9']
    source = string.digits + string.ascii_letters
    return random.choice(source)


# 随机选取一种颜色
def gene_color():
    soure = [(0, 0, 0), (255, 130, 195), (255, 0, 0), (0, 255, 0), (0, 0, 255), (106, 90, 205), (255, 255, 0),
             (184, 134, 11)]
    return random.sample(soure, 1)


# 绘制干扰线
def gene_line(draw, width, height):
    line_color = gene_color()[0]
    begin = (random.randint(0, width), random.randint(0, height))
    end = (random.randint(0, width), random.randint(0, height))
    draw.line([begin, end], fill=line_color, width=1)


# 生成验证码
def gene_code(loader):
    width, height = size  # 宽和高
    image = Image.new('RGB', (width, height), bgcolor)  # 创建图片
    font = ImageFont.truetype(font_path, 40)  # 验证码的字体
    draw = ImageDraw.Draw(image)  # 创建画笔
    text_list = []

    for i in range(number):
        text = gene_text()  # 获取一个字符
        text_list.append(text)
        color = gene_color()  # 获取一种颜色
        font_width, font_height = (font.getsize(text))
        # print(color[0],text)
        draw.text((10 + i * 35, 0), text, font=font, fill=color[0])  # 填充字符

    for i in range(nois):
        x = random.randint(0, width - 5)
        y = random.randint(0, height - 5)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), 'black', 'black')  # 填充噪点
    if draw_line:
        gene_line(draw, width, height)
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 滤镜
    postfix = str(".png")
    loader.set_description(''.join(text_list))
    path = filename + ''.join(text_list) + postfix
    image.save(path)


if __name__ == "__main__":
    loader = tqdm(range(20000))
    for i in loader:
        gene_code(loader)
