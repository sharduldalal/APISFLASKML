import os
from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import face_recognition
import os
from cv2 import cv2
import pickle

UPLOAD_FOLDER = r'C:\Users\shard\Downloads\Face Detection Model\Unknown Faces'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return '''
                    <!doctype html>
                    <title>Upload new File</title>
                    <h1>File Uploaded</h1>
                    '''

            
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
@app.route('/predict')
def run_script():
    li = []

    KNOWN_FACES_DIR = r'C:\Users\shard\Downloads\Face Detection Model\Known Faces'
    UNKNOWN_FACES_DIR = r'C:\Users\shard\Downloads\Face Detection Model\Unknown Faces'
    TOLERANCE = 0.6
    FRAME_THICKNESS = 3
    FONT_THICKNESS = 2
    MODEL = 'cnn' 

    def name_to_color(name):

        color = [(ord(c.lower())-97)*8 for c in name[:3]]
        return color

    print('Loading known faces...')
    known_faces = []
    known_names = []

    for name in os.listdir(KNOWN_FACES_DIR):
        print(name)
        for filename in os.path.join(KNOWN_FACES_DIR, "\\{name}"):
            print(filename)
            image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}\\{name}')
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)
    print('Processing unknown faces...')

    for filename in os.listdir(UNKNOWN_FACES_DIR):
        print(f'Filename {filename}', end='')
        image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}\\{filename}')
        locations = face_recognition.face_locations(image, model=MODEL)
        encodings = face_recognition.face_encodings(image, locations)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        print(f', found {len(encodings)} face(s)')
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            match = None
            if True in results: 
                match = known_names[results.index(True)]
                print(f'{match} this is a match')
                print(f' - {match} from {results}')
                li.append(match)
    
    return  render_template('show.html', li = li)