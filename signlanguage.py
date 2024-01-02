import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import nanocamera as nano
#import tensorflow as tf
#import tensorrt as trt
#import ctypes

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=2,
):
    return (
        "nrguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "

        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
     )

mp_holistic = mp.python.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities

def mediapipe_detection(image, model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color conversion BGR 2 RGB
	image.flags.writeable = False # image is not writeable anymore
	results = model.process(image) # make prediction
	image.flags.writeable = True # image is now writeable again
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # second color conversion
	return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}
print(label_map)

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

res = model.predict(X_test)
actions[np.argmax(res[4])]
actions[np.argmax(y_test[4])]

model.load_weights('action.h5')

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

multilabel_confusion_matrix(ytrue, yhat)
accuracy_score(ytrue, yhat)

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

sequence = []
sentence = []
threshold = 0.8

cap = nano.Camera(flip=2, width=640, height=480, fps=30)
#cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

print('CSI Camera ready? - ', cap.isReady())

frame_skip = 0  # Number of frames to skip
frame_counter = 0  # Initialize frame counter

res = np.zeros(actions.shape[0])

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
	while cap.isReady():

		# Read feed
		frame = cap.read()
		ret = cap.read()

		# Make detections
		image, results = mediapipe_detection(frame, holistic)
		print(results)

		# Draw landmarks
		draw_styled_landmarks(image, results)

		# Prediction logic
		keypoints = extract_keypoints(results)
		#         sequence.insert(0,keypoints)
		#         sequence = sequence[:30]
		sequence.append(keypoints)
		sequence = sequence[-30:]

		if len(sequence) == 30:
			res = model.predict(np.expand_dims(sequence, axis=0))[0]
			print(actions[np.argmax(res)])
		 
		    
		# Viz logic
		if len(res) == len(actions) and max(res) > threshold:
            		action_index = np.argmax(res)
            		action = actions[action_index]

            		if len(sentence) > 0:
                		if action != sentence[-1]:
                    			sentence.append(action)
            		else:
                		sentence.append(action)
		"""
		if res[np.argmax(res)] > threshold: 
			if len(sentence) > 0: 
				if actions[np.argmax(res)] != sentence[-1]:
					sentence.append(actions[np.argmax(res)])
			else:
				sentence.append(actions[np.argmax(res)])
		"""
		if len(sentence) > 5: 
			sentence = sentence[-5:]
		
		# Viz probabilities
		image = prob_viz(res, actions, image, colors)
		    
		cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
		cv2.putText(image, ' '.join(sentence), (3,30), 
			       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

		# Show to screen
		cv2.imshow('OpenCV Feed', image)

		# Break gracefully
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

"""
			if cv2.waitKey(10) & 0xFF == ord('t'):
				for action in actions:
					# Loop through sequences aka videos
					for sequence in range(no_sequences):
					    	# Loop through video length aka sequence length
						for frame_num in range(sequence_length):

							# Read feed
							frame = cap.read()
							
							
							if frame_counter % (frame_skip + 1) == 0:
								# Make detections
								image, results = mediapipe_detection(frame, holistic)
								#print(results)

								# Draw landmarks
								draw_styled_landmarks(image, results)
							
							# NEW Apply wait logic
							if frame_num == 0: 
							    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
								       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
							    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
								       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
							    # Show to screen
							    cv2.imshow('OpenCV Feed', image)
							    cv2.waitKey(4000)
							else: 
							    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
								       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
							    # Show to screen
							    cv2.imshow('OpenCV Feed', image)
							
							# NEW Export keypoints
							keypoints = extract_keypoints(results)
							npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
							np.save(npy_path, keypoints)
"""

cap.release()
cv2.destroyAllWindows()
del cap

result_test = extract_keypoints(results)
print(result_test)

#np.save('0', result_test)
#np.load('0.npy')
