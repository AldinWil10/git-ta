from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('keras_model.h5')

# Load the class labels
with open('labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the camera
cap = cv2.VideoCapture(0)

# Set the camera resolution
cap.set(3, 640)
cap.set(4, 480)

# Set the font for displaying the class name
font = cv2.FONT_HERSHEY_SIMPLEX

def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region
            face_img = frame[y:y+h, x:x+w]

            # Preprocess the face image
            face_img = cv2.resize(face_img, (224, 224))
            face_img = np.expand_dims(face_img, axis=0)
            face_img = face_img / 255.0

            # Predict the class probabilities
            pred_probs = model.predict(face_img)[0]
            class_idx = np.argmax(pred_probs)
            class_prob = pred_probs[class_idx]

            # Get the class name and display it on the image
            class_name = class_names[class_idx]
            if class_prob*100 < 70:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                text = '{}: {:.2f}%'.format('Unknown', class_prob * 100)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(frame, class_name, (x, y - 10), font, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = '{}: {:.2f}%'.format(class_name, class_prob * 100)
                cv2.putText(frame, text, (x, y + h + 30), font, 0.5, (0, 255, 0), 1)

        # Display the resulting frame
        # cv2.imshow('Face Detection', frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # Exit if the 'q' key is pressed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # Release the capture and close the window
# cap.release()
# cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
