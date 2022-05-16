import os
import random
from tqdm import tqdm



def split():
    files = os.walk('./dataset/images')
    images = []
    for _, _, f in files:
        images = f
    os.makedirs('./dataset', exist_ok=True)
    file_train = open('./dataset/train.txt', 'a+')
    file_val = open('./dataset/val.txt', 'a+')
    file_test = open('./dataset/test.txt', 'a+')
    train_rate, val_rate = 0.7, 0.2
    train_size = int(len(images) * train_rate)
    val_size = int(len(images) * val_rate)
    train_set = random.sample(images, train_size)
    for img in train_set:
        images.remove(img)
    val_set = random.sample(images, val_size)
    for img in val_set:
        images.remove(img)
    test_set = images
    loader = tqdm(train_set)
    for img in loader:
        file_train.writelines('{}\t{}\n'.format(img, img[:4]))
    loader = tqdm(val_set)
    for img in loader:
        file_val.writelines('{}\t{}\n'.format(img, img[:4]))
    loader = tqdm(test_set)
    for img in loader:
        file_test.writelines('{}\t{}\n'.format(img, img[:4]))


if __name__ == '__main__':
    split()
