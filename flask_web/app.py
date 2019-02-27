# flask_web/app.py
from flask import Flask
from flask import render_template, request

app = Flask(__name__)


@app.route('/')
def upload():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        return 'file uploaded successfully'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
