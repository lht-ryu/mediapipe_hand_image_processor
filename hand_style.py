from typing import Mapping, Tuple

from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark

RADIUS: int = 5
RED = (48, 48, 255)
GREEN = (48, 255, 48)
BLUE = (192, 101, 21)
YELLOW = (0, 204, 255)
GRAY = (128, 128, 128)
PURPLE = (128, 64, 128)
PEACH = (180, 229, 255)
WHITE = (224, 224, 224)
CYAN = (192, 255, 48)
MAGENTA = (192, 48, 255)

ORANGE = (30, 101, 198)
Light_WHITE = (247, 229, 180)
CYAN2 = (244, 204, 16)

# Hands
THICKNESS_WRIST_MCP = 3
THICKNESS_FINGER = 2
THICKNESS_DOT = -1

# Hand landmarks
PALM_LANDMARKS = (HandLandmark.WRIST, HandLandmark.THUMB_CMC,
                  HandLandmark.INDEX_FINGER_MCP,
                  HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP,
                  HandLandmark.PINKY_MCP)
THUMP_LANDMARKS = (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP,
                   HandLandmark.THUMB_TIP)
INDEX_FINGER_LANDMARKS = (HandLandmark.INDEX_FINGER_PIP,
                          HandLandmark.INDEX_FINGER_DIP,
                          HandLandmark.INDEX_FINGER_TIP)
MIDDLE_FINGER_LANDMARKS = (HandLandmark.MIDDLE_FINGER_PIP,
                           HandLandmark.MIDDLE_FINGER_DIP,
                           HandLandmark.MIDDLE_FINGER_TIP)
RING_FINGER_LANDMARKS = (HandLandmark.RING_FINGER_PIP,
                         HandLandmark.RING_FINGER_DIP,
                         HandLandmark.RING_FINGER_TIP)
PINKY_FINGER_LANDMARKS = (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP,
                          HandLandmark.PINKY_TIP)
HAND_LANDMARK_STYLE = {
    PALM_LANDMARKS:
        DrawingSpec(
            color=PEACH, thickness=THICKNESS_DOT, circle_radius=RADIUS),
    THUMP_LANDMARKS:
        DrawingSpec(
            color=Light_WHITE, thickness=THICKNESS_DOT, circle_radius=RADIUS),
    INDEX_FINGER_LANDMARKS:
        DrawingSpec(
            color=PURPLE, thickness=THICKNESS_DOT, circle_radius=RADIUS),
    MIDDLE_FINGER_LANDMARKS:
        DrawingSpec(
            color=CYAN2, thickness=THICKNESS_DOT, circle_radius=RADIUS),
    RING_FINGER_LANDMARKS:
        DrawingSpec(
            color=GREEN, thickness=THICKNESS_DOT, circle_radius=RADIUS),
    PINKY_FINGER_LANDMARKS:
        DrawingSpec(
            color=ORANGE, thickness=THICKNESS_DOT, circle_radius=RADIUS),
}

# Hands connections
HAND_CONNECTION_STYLE = {
    hands_connections.HAND_PALM_CONNECTIONS:
        DrawingSpec(color=GRAY, thickness=THICKNESS_WRIST_MCP),
    hands_connections.HAND_THUMB_CONNECTIONS:
        DrawingSpec(color=Light_WHITE, thickness=THICKNESS_FINGER),
    hands_connections.HAND_INDEX_FINGER_CONNECTIONS:
        DrawingSpec(color=PURPLE, thickness=THICKNESS_FINGER),
    hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS:
        DrawingSpec(color=CYAN2, thickness=THICKNESS_FINGER),
    hands_connections.HAND_RING_FINGER_CONNECTIONS:
        DrawingSpec(color=GREEN, thickness=THICKNESS_FINGER),
    hands_connections.HAND_PINKY_FINGER_CONNECTIONS:
        DrawingSpec(color=ORANGE, thickness=THICKNESS_FINGER)
}


def hand_landmarks_style() -> Mapping[int, DrawingSpec]:

    hand_landmark_style = {}
    for k, v in HAND_LANDMARK_STYLE.items():
        for landmark in k:
            hand_landmark_style[landmark] = v
    return hand_landmark_style


def hand_connections_style() -> Mapping[Tuple[int, int], DrawingSpec]:
    """Returns the default hand connections drawing style.

  Returns:
      A mapping from each hand connection to its default drawing spec.
  """
    hand_connection_style = {}
    for k, v in HAND_CONNECTION_STYLE.items():
        for connection in k:
            hand_connection_style[connection] = v
    return hand_connection_style
