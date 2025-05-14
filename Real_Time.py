
import numpy as np
import pickle
import cv2
from math import cos, sin
import mediapipe as mp


with open('pitch_model.pkl', 'rb') as f:
    pitch_model = pickle.load(f)
with open('yaw_model.pkl', 'rb') as f:
    yaw_model = pickle.load(f)
with open('roll_model.pkl', 'rb') as f:
    roll_model = pickle.load(f)


def draw_axis(img, pitch,yaw,roll, tdx=None, tdy=None, size = 100):
    # """
    # Draws 3D coordinate axes on an image based on head pose angles: pitch, yaw, and roll.

    # Args:
    #     img: The image on which to draw.
    #     pitch: Head pitch angle (up/down movement).
    #     yaw: Head yaw angle (left/right movement).
    #     roll: Head roll angle (tilting sideways).
    #     tdx: x-coordinate for the origin of the axes (optional).
    #     tdy: y-coordinate for the origin of the axes (optional).
    #     size: Length of the axes lines.

    # Returns:
    #     The image with the drawn axes.
    # """

    # Invert yaw to align coordinate system with image display
    yaw = -yaw

    # Set center of drawing (origin point) to tdx, tdy if provided; else, center of the image
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2


    # ---- Calculate Endpoints of Each Axis ---- #

    # X-axis (Red): Right direction
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-axis (Green): Down direction
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-axis (Blue): Out of the screen direction
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    # ---- Draw Each Axis Line ---- #

    # cv2.line function
    # Draw a line on the image:
    # - img: image to draw on
    # - (int(tdx), int(tdy)): starting point (origin of axes)
    # - (int(x), int(y)): ending point (?-axis direction)
    # - (0, 0, 0): color in BGR format
    # - x : thickness of the line

    # Draw X-axis in RED from center to (x1, y1)
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    # Draw Y-axis in GREEN from center to (x2, y2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    # Draw Z-axis in BLUE from center to (x3, y3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    # Return the final image
    return img



# function to detect landmarks for an image
def detect_landmarks(img):
  # set up face mesh module
  face_module = mp.solutions.face_mesh

  # initialize arrays for features
  X = []
  y = []

  # create face mesh object
  with face_module.FaceMesh (static_image_mode=True)as face_mesh:
      # process image for the face_mesh model
      results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      # check if face landmarks were detected
      if results.multi_face_landmarks:
          # get face landmarks for all detected faces
          for face_landmarks in results.multi_face_landmarks:
              # initialize array for this set of landmarks
              shape = img.shape
              for landmark in face_landmarks.landmark:
                  # get landmark coordinates and add them to X and y lists
                  relative_x = int(landmark.x * shape[1])
                  relative_y = int(landmark.y * shape[0])
                  X.append(relative_x)
                  y.append(relative_y)

  X = np.array([X])
  y = np.array([y])

  # Normalizing features
  Nose_centered_X = X - X[:,1].reshape(-1,1)
  Nose_centered_y = y - y[:,1].reshape(-1,1)

  X_171 = X[:,171]
  X_10 = X[:,10]
  y_171 = y[:,171]
  y_10 = y[:,10]
  distance = np.linalg.norm(np.array((X_10,y_10)) - np.array((X_171, y_171)),axis = 0).reshape(-1,1)
  Norm_X = Nose_centered_X / distance
  Norm_Y = Nose_centered_y / distance

  nose_x = X[:,1]
  nose_y = y[:,1]

  features = np.hstack([Norm_X,Norm_Y])

  return features, nose_x, nose_y
def model_predict(pitch_model, yaw_model, roll_model, features):
  # making predictions
  pitch = pitch_model.predict(features)
  yaw = yaw_model.predict(features)
  roll = roll_model.predict(features)
  return pitch, yaw, roll
def perform_prediction(pitch_model, yaw_model, roll_model, img):
  features, nose_x, nose_y = detect_landmarks(img)
  pitch, yaw, roll = model_predict(pitch_model, yaw_model, roll_model, features)
  frame = draw_axis(img, pitch, yaw, roll, nose_x, nose_y)
  return frame,pitch, yaw, roll

cap = cv2.VideoCapture(0)
while cap.isOpened():

        try:

        # Read the next frame from the video
            ret, frame = cap.read()
            if not ret:
              print("Error: Failed to read frame from webcam.")
              break
            frame = cv2.flip(frame, 1)
            frame, pitch, yaw, roll = perform_prediction(pitch_model, yaw_model, roll_model, frame)

            # Add a border around the whole frame
            height, width = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 0), 10)

        # Add a square border around the face
            features, nose_x, nose_y = detect_landmarks(frame)
            if nose_x.size > 0 and nose_y.size > 0:
                nose_x, nose_y = int(nose_x[0]), int(nose_y[0])
                square_size = 300  # Adjust the size of the square as needed
                top_left = (nose_x - square_size // 2, nose_y - square_size // 2)
                bottom_right = (nose_x + square_size // 2, nose_y + square_size // 2)
                cv2.rectangle(frame, top_left, bottom_right, (255,0, 255), 2)

                # Display the frame with the drawn axes
                cv2.putText(frame, f'Yaw: {float(yaw)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f'Pitch: {float(pitch)}', (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f'Roll: {float(roll)}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) == ord('q'):
              break
        except:
          pass


cap.release()
cv2.destroyAllWindows()
# Release the video capture and close cv2 windows