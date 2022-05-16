from shutil import copyfile
import os

os.makedirs('./dataset/test/', exist_ok=True)
images = []
with open('./dataset/test.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        images.append(line.split('\t')[0])
    for img in images:
        copyfile('./dataset/images/' + img, './dataset/test/' + img)
