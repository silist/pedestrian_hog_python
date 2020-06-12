# -*- encoding: utf-8 -*-

"""
@File    :   app.py
@Time    :   2020/06/10 15:44:04
@Author  :   silist
@Version :   1.0
@Desc    :   Flask DEMO for HOG+SVM.
"""

import os
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify, abort
from werkzeug.utils import secure_filename

from hog.loader.config_loader import Config
from utils import get_feature, load_model, get_hog, get_prediction

# 上传目录
UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """接收前端上传的图片文件，返回地址"""
    if request.method == 'POST':
        file_data = request.files['file_data']  # FileStorge格式
        if file_data:
            file_name = secure_filename(file_data.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            file_data.save(file_path)
            return jsonify({'file_path': file_path})
        else:
            # Return error code 400
            return jsonify(message='fileuploaderror'), 400

@app.route('/predict', methods=['POST'])
def predict_image():
    """接收前端上传的图片地址，返回预测标签"""
    if request.method == 'POST':
        file_path = request.form['file_path']
        try:
            label = predict_label(file_path)
            return jsonify({'file_path': file_path, 'label': label})
        except Exception as e:
            return jsonify(message=str(e)), 400

def predict_label(img_path):
    feature = get_feature(cfg, hog, img_path)
    label = get_prediction(model, feature)
    return label

if __name__ == '__main__':
    cfg_path = r'./hog/config/hog_svm_inria_pc.yml'
    model_path = r'./hog/pkl/hog_inria_svm_linear_interp_none_flask.pkl'

    # 加载HOG配置文件及模型
    cfg = Config(cfg_path)
    model = load_model(model_path)
    hog = get_hog(cfg)

    print('Finish loading config and model.')

    # Run Flask
    app.run(debug=True)
