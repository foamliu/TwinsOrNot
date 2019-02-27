# flask_web/app.py
import time
import os
from flask import Flask
from flask import render_template, request
from werkzeug.utils import secure_filename
from utils import compare, ensure_folder

app = Flask(__name__)


@app.route('/')
def upload():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            start = time.time()
            ensure_folder('static')
            file1 = request.files['file1']
            filename_1 = secure_filename(file1.filename)
            filename_1 = os.path.join('static', filename_1)
            file1.save(filename_1)
            file2 = request.files['file2']
            filename_2 = secure_filename(file1.filename)
            filename_2 = os.path.join('static', filename_2)
            file2.save(filename_2)
            theta, is_same = compare(filename_1, filename_2)
            elapsed = time.time() - start
            message = '两张照片是否同一个人: {}, 角度: {}, 时间: {} 秒。'.format(is_same, theta, elapsed)
        except ValueError:
            message = '对不起，没有检测到人类的脸'
        return render_template('show.html', message=message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, threaded=True)
