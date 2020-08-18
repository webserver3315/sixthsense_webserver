from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from device import Device
from yolov5.detect_photo_version2 import get_detected_image_from_photo
import os
import pickle

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'sunshine!123'
app.debug = True
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

devices = dict()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/image/<device_id>', methods=['POST'])
def update_image(device_id):
    if not device_id:
        return 400, "no device id"

    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect('/')
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect('/')
    if file and allowed_file(file.filename):
        
        extension = os.path.splitext(file.filename)[1]
        print(extension)
        filename = device_id + '_' + request.form['timestamp'] + extension
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        obj_list = None
        if device_id not in devices:
            devices[device_id] = Device(device_id)
            obj_list = devices[device_id].new_frame(filepath, request.form['timestamp'])
        else:
            device = devices[device_id]
            obj_list = device.new_frame(filepath, request.form['timestamp'])
        return pickle.dumps(obj_list)
