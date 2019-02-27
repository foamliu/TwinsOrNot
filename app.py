# flask_web/app.py
import os
import time

from flask import Flask
from flask import render_template, request
from werkzeug.utils import secure_filename

from utils import compare, ensure_folder, FaceNotFoundError

app = Flask(__name__, static_url_path="", static_folder="static")


@app.route('/')
def upload():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        start = time.time()
        ensure_folder('static')
        file1 = request.files['file1']
        filename_1 = secure_filename(file1.filename)
        full_path_1 = os.path.join('static', filename_1)
        file1.save(full_path_1)
        file2 = request.files['file2']
        filename_2 = secure_filename(file2.filename)
        full_path_2 = os.path.join('static', filename_2)
        file2.save(full_path_2)

        try:
            theta, is_same = compare(full_path_1, full_path_2)
            elapsed = time.time() - start
            message = '两张照片是否同一个人: {}, 角度: {}, 时间: {} 秒。'.format(is_same, theta, elapsed)
        except FaceNotFoundError as err:
            message = '对不起，[{}] 图片中没有检测到人类的脸。'.format(err)

        return render_template('show.html', message=message, filename_1=filename_1, filename_2=filename_2)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, threaded=True)
