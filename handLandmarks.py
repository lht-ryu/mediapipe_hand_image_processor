import cv2
import mediapipe.python.solutions as solutions
import mediapipe.python.solutions.drawing_utils as drawing_utils
from mediapipe.framework.formats import landmark_pb2
import hand_style
import numpy as np

##################################################

mp_draw = drawing_utils

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

#hand_landmarks_style = mp_draw.DrawingSpec(color=(255,2,0),  circle_radius=4)
#hand_connections_style = mp_draw.DrawingSpec(color=(0,2,255), thickness= 2,)


global ihand_landmarks_style
global ihand_connections_style

ihand_landmarks_style = hand_style.hand_landmarks_style()
ihand_connections_style = hand_style.hand_connections_style()

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  #annotated_image = rgb_image
  annotated_image = np.copy(rgb_image)

  #rom annotator.mediapipe_hand import hand_style

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    mp_draw.draw_landmarks(
      image=annotated_image,
      landmark_list=hand_landmarks_proto,
      connections=solutions.hands.HAND_CONNECTIONS,
      landmark_drawing_spec=ihand_landmarks_style,
      connection_drawing_spec=ihand_connections_style
    )

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    '''cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
'''

  return annotated_image
