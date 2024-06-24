from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import io
import json
import numpy as np
from PIL import Image
from keras.utils import load_img, img_to_array
from keras.models import load_model
from pymongo import MongoClient

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load skin classes from the JSON file
with open('class_indices.json', 'r') as f:
    SKIN_CLASSES = json.load(f)

# Reverse the dictionary to get class labels by index
SKIN_CLASSES = {v: k for k, v in SKIN_CLASSES.items()}

def connect_to_db():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['skin-disease-db']
    return db

def register_user(username, email, password):
    db = connect_to_db()
    user = {'username': username, 'email': email, 'password': password}
    users = db['users']
    result = users.insert_one(user)
    return str(result.inserted_id)

def authenticate_user(username, password):
    db = connect_to_db()
    users = db['users']
    user = users.find_one({'username': username})
    if user and user['password'] == password:
        return True
    return False

login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    pass

@login_manager.user_loader
def user_loader(username):
    db = connect_to_db()
    users = db['users']
    user = users.find_one({'username': username})
    if user:
        u = User()
        u.id = user['username']
        return u
    return None

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not authenticate_user(username, password):
            error = 'Invalid username or password. Please try again.'
            return render_template('signin.html', error=error)
        else:
            user = User()
            user.id = username
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm']

        if not username or not email or not password or not confirm_password:
            return render_template('error.html', message='Please fill in all fields.')
        if password != confirm_password:
            return render_template('error.html', message='Passwords do not match.')

        register_user(username, email, password)
        return redirect(url_for('signin'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')

def findMedicine(pred):
    # Add your medicine mapping logic here
    medicine_map = {
        0: "fluorouracil",
        1: "Aldara",
        2: "Tetracycline, Minocycline and Doxycycline",
        3: "fluorouracil",
        4: "fluorouracil (5-FU)",
        5: "fluorouracil",
        6: "fluorouracil",
        7: "fluorouracil",
        8: "fluorouracil",
        9: "fluorouracil",
        10: "fluorouracil"
    }
    return medicine_map.get(pred, "No medicine available")

@app.route('/detect', methods=['GET', 'POST'])
@login_required  # Ensure the user is logged in to access this route
def detect():
    if request.method == 'POST':
        try:
            file = request.files['file']
        except KeyError:
            return jsonify({
                'error': 'No file part in the request',
                'code': 'FILE',
                'message': 'file is not valid'
            }), 400

        imagePil = Image.open(io.BytesIO(file.read()))
        imageBytesIO = io.BytesIO()
        imagePil.save(imageBytesIO, format='JPEG')
        imageBytesIO.seek(0)

        path = imageBytesIO
        model = load_model('model.keras')  # Load model from .keras file
        img = load_img(path, target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape((1, 224, 224, 3))
        img = img / 255
        prediction = model.predict(img)
        pred = np.argmax(prediction)
        if pred not in SKIN_CLASSES:
            return render_template('error.html', message='Invalid prediction')
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]
        accuracy = round(accuracy * 100, 2)
        medicine = findMedicine(pred)

        json_response = {
            "detected": False if pred == 2 else True,
            "disease": disease,
            "accuracy": accuracy,
            "medicine": medicine,
            "img_path": file.filename,
        }

        session['detection_result'] = json_response
        return redirect(url_for('detected'))
    else:
        return render_template('detect.html')

@app.route('/detected', methods=['GET'])
@login_required
def detected():
    detection_result = session.get('detection_result', None)
    if not detection_result:
        return render_template('error.html', message='No detection result available.')
    return render_template('detected.html', result=detection_result)

if __name__ == "__main__":
    app.run(debug=True, port=3000)