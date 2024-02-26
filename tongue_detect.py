import dlib
from facenet_pytorch import MTCNN
import numpy as np
import cv2

def detect_tongue(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Initialize the MTCNN face detection model
    mtcnn = MTCNN(keep_all=True)

    # Detect faces in the image
    boxes, probs = mtcnn.detect(image)

    if boxes is None:
        print("No face detected in the image.")
        # Continue with the whole image if no face is detected
        face_roi = image
    else:
        # Assuming the first detected face is the correct one
        face_box = boxes[0].astype(int)
        
        # Extract the region of interest (ROI) around the detected face
        face_roi = image[face_box[1]:face_box[3], face_box[0]:face_box[2]]

    # Load the pre-trained facial landmark predictor from dlib
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    # Convert the face ROI to grayscale for facial landmark detection
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Detect facial landmarks
    landmarks = predictor(gray_face, dlib.rectangle(0, 0, face_roi.shape[1], face_roi.shape[0]))

    # Extract the tongue region (assuming landmarks 54-59 represent the tongue)
    tongue_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(54, 60)])

    # Create a mask for the tongue region
    mask = np.zeros_like(gray_face)
    cv2.fillPoly(mask, [tongue_landmarks], 255)

    # Display the detected tongue
    tongue = cv2.bitwise_and(gray_face, gray_face, mask=mask)
    cv2.imshow("Detected Tongue", tongue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()