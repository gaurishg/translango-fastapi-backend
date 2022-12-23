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
from google.cloud import translate
from collections import defaultdict
from typing import List, Dict, Tuple

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



translation_map = defaultdict(dict)


def translate_text(text="Hello, world!", target_lang="ja", source_lang=None, project_id="translango-gaurish"):

    client = translate.TranslationServiceClient()
    location = "global"
    parent = f"projects/{project_id}/locations/{location}"
    request_dict = {
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",
            "target_language_code": target_lang
            }
    if source_lang:
        request_dict["source_language_code"] = source_lang
    response = client.translate_text(
        request=request_dict
    )

    # for translation in response.translations:
    #    print("Translated text: {}".format(translation.translated_text))
    return list(response.translations)[0].translated_text

def add_translations_to_detections(detections: List[Dict], target_lang='ja'):
    for detection in detections:
        if detection['name'] not in translation_map or target_lang not in translation_map[detection['name']]:
            translation_map[detection['name']][target_lang] = translate_text(detection['name'], target_lang)
        detection['translation'] = translation_map[detection['name']][target_lang]

 
@app.route('/', methods=['GET'])
def home():
    image_url = request.args.get('url')
    target_lang = request.args.get('target_lang') or 'ja'
    if image_url is None:
        return render_template('index.html')
    else:
        detections = yolov7.translango.translango_detect_from_url(image_url)
        add_translations_to_detections(detections, target_lang)
        response = {
                'error': 'no',
                'error_msg': 'no error',
                'detections': detections
                }
        response_json = json.dumps(response)
        return response_json


@app.route('/', methods=['POST'])
def home_post():
    image_url = request.form.get('url', None)
    target_lang = request.form.get('target_lang', 'en')
    if image_url is None:
        response = {
            'error': 'yes',
            'error_msg': 'provide a valid image url',
            'detections': [],
                }
        return json.dumps(response)
    else:
        detections = yolov7.translango.translango_detect_from_url(image_url)
        add_translations_to_detections(detections, target_lang)
        response = {
                'error': 'no',
                'error_msg': 'none',
                'detections': detections,
                }
        return json.dumps(response)


@app.route('/text-translate', methods=['GET'])
def text_translate():
    text = request.args.get('text')
    text = text if text is not None else 'hello'
    # source_lang = request.args.get('source_lang')
    # source_lang = source_lang if source_lang is not None else 'en'
    target_lang = request.args.get('target_lang')
    target_lang = target_lang if target_lang is not None else 'ja'
    source_lang = request.args.get('source_lang', None)
    if text not in translation_map or target_lang not in translation_map[text]:
        translation_map[text][target_lang] = translate_text(text, target_lang, source_lang)
    return translation_map[text][target_lang]

@app.route('/text-translate', methods=['POST'])
def text_translate_post():
    text = request.form.get('text', 'hello')
    target_lang = request.form.get('target_lang', 'ja')
    source_lang = request.form.get('source_lang', None)
    if text not in translation_map or target_lang not in translation_map[text]:
        translation_map[text][target_lang] = translate_text(text, target_lang, source_lang)
    return translation_map[text][target_lang]

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
    # print(request)
    target_lang = request.form['target_lang']
    # target_lang = target_lang if target_lang is not None else 'en'
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        detections = yolov7.translango.translango_detect(array)
        add_translations_to_detections(detections, target_lang)
        # for detection in detections:
        #     if detection['name'] not in translation_map or target_lang not in translation_map[detection['name']]:
        #         translation_map[detection['name']][target_lang] = translate_text(detection['name'], target_lang)
        #     detection['translation'] = translation_map[detection['name']][target_lang]
        #     print(f"{detection['name']} -> {detection['translation']}")

        complete_response = {
            'error': 'no',
            'error_msg': 'no error',
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
            'error_msg': 'Allowed image types are - png, jpg, jpeg, gif',
            'detections': None
        }
        # return redirect(request.url)
        return json.dumps(error_response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
