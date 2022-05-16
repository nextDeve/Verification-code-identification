from tkinter.tix import IMAGE
import cv2
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import pandas as pd
import os
import PIL.Image as Image
from pathlib import Path
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import io
import flask
import json
import base64
from flask import send_file
from predict import predict_res

source = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
          'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
          'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def del_files(test_path, set_num):
    num = 0
    for root, dirs, files in os.walk(test_path):
        files_size = len(files)
        for name in files:
            if name.endswith(".png"):  # 指定要删除的格式
                if (num < files_size - set_num):
                    os.remove(os.path.join(root, name))
                    num += 1
                else:
                    break


def noise_remove_cv2(image_name, k):
    def calculate_noise_count(img_obj, w, h):
        count = 0
        width, height = img_obj.shape
        for _w_ in [w - 1, w, w + 1]:
            for _h_ in [h - 1, h, h + 1]:
                if _w_ > width - 1:
                    continue
                if _h_ > height - 1:
                    continue
                if _w_ == w and _h_ == h:
                    continue
                if img_obj[_w_, _h_] < 255:
                    count += 1
        return count

    gray_img = cv2.cvtColor(image_name, cv2.COLOR_BGR2GRAY)  # 灰度化
    thresh, gray_img = cv2.threshold(gray_img, 254, 255, cv2.THRESH_BINARY)  # 二值化

    w, h = gray_img.shape
    for _w in range(w):
        for _h in range(h):
            if _w == 0 or _h == 0:
                gray_img[_w, _h] = 255
                continue
            # 计算邻域pixel值小于255的个数
            pixel = gray_img[_w, _h]
            if pixel == 255:
                continue
            if calculate_noise_count(gray_img, _w, _h) < k:  # 通过阈值判断噪点
                gray_img[_w, _h] = 255
    return gray_img


def function(imagefile):
    os.makedirs('./extradata/flask/split/', exist_ok=True)
    image = cv2.imread(imagefile)
    noise_img = noise_remove_cv2(image, 4)  # 八邻域降噪
    noise_img = cv2.medianBlur(noise_img, 3)  # 中值滤波
    thresh, noise_img = cv2.threshold(noise_img, 127, 255, cv2.THRESH_BINARY)  # 二值化
    contours, hierarchy = cv2.findContours(noise_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找最外层轮廓
    dfContourShape = pd.DataFrame(columns=('X', 'Y', 'W', 'H'))
    for i in range(len(contours)):
        cv2.drawContours(image, contours, i, (0, 255, 0), 1)  # i 表示绘制第i条轮廓
        x, y, w, h = cv2.boundingRect(contours[i])  # 用一个最小的矩形，把找到的形状包起来
        dfContourShape = dfContourShape.append([{'X': x, 'Y': y, 'W': w, 'H': h}], ignore_index=True)

    result = []
    for i in range(len(dfContourShape)):
        x = dfContourShape['X'][i]
        y = dfContourShape['Y'][i]
        w = dfContourShape['W'][i]
        h = dfContourShape['H'][i]
        wMax = max(dfContourShape['W'])
        wMin = min(dfContourShape['W'])
        if len(dfContourShape) == 1:  # 将轮廓四等分
            boxLeft = np.int0([[x, y], [x + w / 4, y], [x + w / 4, y + h], [x, y + h]])
            boxMidLeft = np.int0([[x + w / 4, y], [x + w * 2 / 4, y], [x + w * 2 / 4, y + h], [x + w / 4, y + h]])
            boxMidRight = np.int0(
                [[x + w * 2 / 4, y], [x + w * 3 / 4, y], [x + w * 3 / 4, y + h], [x + w * 2 / 4, y + h]])
            boxRight = np.int0([[x + w * 3 / 4, y], [x + w, y], [x + w, y + h], [x + w * 3 / 4, y + h]])
            result.extend([boxLeft, boxMidLeft, boxMidRight, boxRight])
    imgs = []
    for i, box in enumerate(result):
        cv2.drawContours(image, [box], 0, (0, 0, 255), 1)
        roi = noise_img[box[0][1]:box[3][1], box[0][0]:box[1][0]]  # 取二值化图片分割后的每一张图片
        roiStd = cv2.resize(roi, (50, 50))  # 将字符图片统一调整为50x50的图片大小
        fileSavePath = './extradata/flask/split/{}.png'.format(i)
        imgs.append(fileSavePath)
        cv2.imwrite(fileSavePath, roiStd)
    return imgs


app = Flask(__name__)  # 实例化，可视为固定格式
app.debug = True  # Flask内置了调试模式，可以自动重载代码并显示调试信息
app.config['JSON_AS_ASCII'] = False  # 解决flask接口中文数据编码问题

# 设置可跨域范围
CORS(app, supports_credentials=True)


def return_img_stream(img_local_path):
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream


@app.route("/templates")
def verify():
    return send_file("verify.html")


# 分割
@app.route("/split", methods=["POST"])
def split():
    os.makedirs("./extradata/flask/save/", exist_ok=True)
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            upload_file = request.files['image']
            file_name = upload_file.filename
            file_name_ = "./extradata/flask/save/" + ''.join(file_name)
            upload_file.save(file_name_)
            img_pathpath__ = function(file_name_)
            img_stream_ = []
            for pp in range(4):
                img_path_ = img_pathpath__[pp]
                img_stream = return_img_stream(img_path_)
                img_stream_.append(img_stream)
            return render_template('verify.html', img_stream_=img_stream_)


#  跳转到html页面显示图片,app.route()为跳转路由
@app.route('/')
def split_main():
    img_stream_ = []
    return render_template('verify.html', img_stream_=img_stream_)


# 预测
@app.route('/predict', methods=["POST"])
def predict():
    predict_result_str = predict_res()
    del_files("./extradata/flask/split/", 0)
    del_files("./extradata/flask/save/", 0)
    return predict_result_str


# #  跳转到html页面显示预测结果
# @app.route('/')
# def predict_main():
#     predict_result_str = []
#     return render_template('verify.html',predict_result_str = predict_result_str)

if __name__ == '__main__':
    # app.run(host, port, debug, options)
    app.run(host="127.0.0.1", port=5000)
