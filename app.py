# flask_web/app.py
import time

from flask import Flask
from flask import render_template, request

from utils import compare, ensure_folder

app = Flask(__name__)


@app.route('/')
def upload():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        start = time.time()
        ensure_folder('static')
        file1 = request.files['file1']
        file1.save('static/img0.png')
        file2 = request.files['file2']
        file2.save('static/img1.png')
        theta, is_same = compare('static/img0.png', 'static/img1.png')
        elapsed = time.time() - start
        message = '两张照片是否同一个人: {}, 角度: {}, 时间: {} 秒。'.format(is_same, theta, elapsed)
        return render_template('show.html', message=message)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6006)
