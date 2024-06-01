
import cv2
import os
from flask import Flask,request,render_template,redirect,session,url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import time
from db.models import User
from flask import Blueprint
from db import models


from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
db = SQLAlchemy()

def find_camera_index():
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

camera_index = find_camera_index()
if camera_index is None:
    raise Exception("No camera found")
else:
    cap = cv2.VideoCapture(camera_index)


def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc://DESKTOP-VO8BRER\\NND:@server/database'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    return app



# import db

#VARIABLES
MESSAGE = "WELCOME  " \
          " Instruction: to register your attendence kindly click on 'a' on keyboard"

#### Defining Flask App




ADMIN_USERNAME = 'harsh'
ADMIN_PASSWORD = 'harsh@123'

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')

#### get a number of total registered users

  

def fetch_registered_users():
    # Assuming you're using SQLAlchemy
    registered_users = models.query.all()
    
    # Extract relevant information from the registered users
    user_data = []
    for user in registered_users:
        user_data.append({
            'id': user.id,
            'username': user.username,
            # Add more fields as needed
        })
    
    return user_data






def totalreg():
    return len(os.listdir('static/faces'))

#### extract the face from an image
def extract_faces(img):
    if img is not None and len(img.shape) == 3:  # Check if the image is not None and has three dimensions
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []


#### Identify face using ML model
def identify_face(face_array):
    model = joblib.load('static/face_recognition_model.pkl')
    resized_face_array = resize_input(face_array)
    return model.predict(resized_face_array)

def resize_input(face_array):
    # Assuming the model was trained on 10x10 color images (RGB)
    expected_size = (10, 10)
    resized_face = cv2.resize(face_array, expected_size)
    flattened_face = resized_face.flatten().reshape(1, -1)  # Shape should be (1, 300) for RGB images
    return flattened_face

def capture_face():
    # Your code to capture a face image using OpenCV
    # Ensure the captured face is in color (RGB) if the model was trained on color images
    pass

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (10, 10))
            faces.append(resized_face.flatten())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l

#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if str(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')
    else:
        print("this user has already marked attendence for the day , but still i am marking it ")
        # with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
        #     f.write(f'\n{username},{userid},{current_time}')


################## ROUTING FUNCTIONS ##############################





@app.route('/admin/login')
def admin_login():
    # Render the admin login page template
    return render_template('admin_login.html')

# Route for handling the admin login form submission
@app.route('/admin/login', methods=['POST'])
def admin_login_submit():
    username = request.form['username']
    password = request.form['password']
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        # Redirect to the admin dashboard or another page upon successful login
        return redirect(url_for('admin_dashboard'))
    else:
        # Handle invalid credentials
        return 'Invalid username or password'

# Route for rendering the admin dashboard
@app.route('/admin/dashboard')
def admin_dashboard():
    # Render the admin dashboard page template
    return render_template('admin_dashboard.html')

@app.route('/view_registered_users')
def view_registered_users():
    # Assuming you have a function to fetch registered users from your database
     # Implement this function
    
    # Render a template to display the list of registered users
    return render_template('registered_users.html')

#### Our main page
@app.route('/')
def home():
    names,rolls,times,l = extract_attendance()
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2, mess = MESSAGE)


#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    ATTENDENCE_MARKED = False
    if 'static/face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'This face is not registered with us , kindly register yourself first'
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = frame[y:y+h, x:x+w]
            identified_person = identify_face(face)[0]
            cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            if cv2.waitKey(1) == ord('a'):
                add_attendance(identified_person)
                current_time_ = datetime.now().strftime("%H:%M:%S")
                print(f"attendence marked for {identified_person}, at {current_time_} ")
                ATTENDENCE_MARKED = True
                break
        if ATTENDENCE_MARKED:
            break

        cv2.imshow('Attendance Check, press "q" to exit', frame)
        cv2.putText(frame, 'hello', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    MESSAGE = 'Attendence taken successfully'
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500 or i == 50:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    return redirect(url_for('home'))

#### Our main function which runs the Flask App

if __name__ == '__main__':
  app.run(debug=True)
#### This function will run when we add a new user
