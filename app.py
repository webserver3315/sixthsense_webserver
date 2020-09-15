from flask import Flask, flash, request, redirect, url_for, send_from_directory, make_response, render_template
from werkzeug.utils import secure_filename
from device import Device
from config import UPLOAD_FOLDER
from accident_cal import accident_percentage
import os
import pickle
import cv2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'sunshine!123'
app.debug = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

devices = dict()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/clear/<device_id>', methods=['GET'])
def clear_device(device_id):
    if not device_id:
        return "no device id", 400
    if device_id not in devices:
        return "no device to delete", 400
    return str(devices.pop(device_id))


@app.route('/image/<device_id>', methods=['POST'])
def update_image(device_id):
    if not device_id:
        return "no device id", 400
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect('/')
    file = request.files['file']
    print(file.filename)
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

@app.route('/percentage/<device_id>', methods=['GET'])
def percentage(device_id):
    matrix = devices[device_id].accident_percentage_matrix
    return max(max(x) for x in matrix if not 0)
    

@app.route('/framed/<device_id>', methods=['GET'])
def framed(device_id):
    if not devices[device_id].framed_images:
        return 400
    retval, buffer = cv2.imencode('.png', devices[device_id].framed_images[-1])
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'
    return response

@app.route('/framed/<device_id>/<idx>', methods=['GET'])
def framed_idx(device_id):
    pass

@app.route('/dashboard/<device_id>', methods=['GET'])
def dashboard(device_id):
    device = devices[device_id]
    
    print(device.objects_list)
    print(accident_percentage(device.objects_list))
    ap = accident_percentage(device.objects_list)
    max_per = [max(x) for x in ap if x]

    max_per = max(max_per) if max_per else 0


    return render_template('dashboard.html', image_url='/framed/'+device_id, size=max_per*400, danger_color=f"{int(min(255, 510*max_per)):02x}{int(min(255, 510-(510*max_per))):02x}00")

@app.route('/uploads/<path:filename>')
def serve_static(filename):
    print(UPLOAD_FOLDER)
    return send_from_directory(UPLOAD_FOLDER, filename)

