#app.py
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from PIL import Image
from io import BytesIO
from tempfile import SpooledTemporaryFile
import numpy as np
import json
import sys
current_path = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(current_path, '..')))
sys.path.append(os.path.abspath(os.path.join(current_path, '..', 'yolov7')))
# print(sys.path)
import yolov7.translango

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file: FileStorage = request.files['file']
    file_stream: SpooledTemporaryFile = file.stream
    image_bytes = file_stream.read()
    image = Image.open(BytesIO(image_bytes))
    array = np.array(image)
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        detections = yolov7.translango.translango_detect(array)
        complete_response = {
            'error': 'no',
            'detections': detections
        }
        response_json = json.dumps(complete_response)
        # flash(response_json)
        # return render_template('index.html', detections=response_json)
        return response_json
    else:
        # flash('Allowed image types are - png, jpg, jpeg, gif')
        print('Allowed image types are - png, jpg, jpeg, gif')
        error_response = {
            'error': 'yes',
            'error_msg': 'Allowed image types are - png, jpg, jpeg, gif'
        }
        # return redirect(request.url)
        return json.dumps(error_response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
