from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, List, Optional, Any
import math
import cv2


class Complexity(Enum):
    """
    Enum representing the complexity level of a filter.
    """
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Detection(NamedTuple):
    """
    define a generic class for xywh format
    """
    x: float
    y: float
    w: float
    h: float
    confidence: float = 0.0
    label: str=""
    keypoints: Optional[list] = None


class Instruction(Enum):
    """
    Enum representing various instructions for user guidance.
    """
    MOVE_LEFT = "Move Left"
    MOVE_RIGHT = "Move Right"
    MOVE_UP = "Tilt Up"
    MOVE_DOWN = "Tilt Down"
    COME_CLOSER = "Move Closer"
    STEP_BACK = "Step Back"
    LEVEL_SHOULDERS = "Level Your Shoulders"
    CHIN_UP = "Raise Your Chin"
    CHIN_DOWN = "Lower Your Chin"
    TILT_HEAD_RIGHT = "Tilt Your Head Right"
    TILT_HEAD_LEFT = "Tilt Your Head Left"
    TURN_HEAD_RIGHT = "Turn Your Head Right"
    TURN_HEAD_LEFT = "Turn Your Head Left"
    STAND_STRAIGHT = "Stand Up Straight"
    READY = "Ready! Shoot!"
    SEARCHING = "Searching for person..."
    FIND_SKY = "Adjust to show more sky in the background"


class BaseFilter(ABC):
    """
    Base class for all filters.
    """
    def __init__(self, name, complexity: Complexity, model=None):
        self._name = name
        self._complexity = complexity
        self._model = model

    @property
    def name(self):
        """Returns the name of the filter"""
        return self._name

    @property
    @abstractmethod
    def description(self):
        """Returns a short description of the filter"""
        pass

    @abstractmethod
    def _calculate_feedback(self, frame, detections):
        pass

    def apply(self, frame):
        detections = self._get_detections(frame)
        return self._calculate_feedback(frame, detections)

    def subject_centered(self, detection: Detection, tolerance=0.05):
        """Checks if the subject is centered by the vertical axis"""
        if detection.x < 0.5 - tolerance:
            return [Instruction.MOVE_RIGHT.value]
        if detection.x > 0.5 + tolerance:
            return [Instruction.MOVE_LEFT.value]
        return []

    def subject_on_third(self, detection: Detection, tolerance=0.05):
        """Checks if the subject is on one of the vertical third lines (0.33 or 0.66)."""
        left_third, right_third = 1 / 3, 2 / 3
        if abs(detection.x - left_third) <= tolerance or abs(detection.x - right_third) <= tolerance:
            return []

        # Guide to the closest third line
        if abs(detection.x - left_third) < abs(detection.x - right_third):
            return [Instruction.MOVE_RIGHT.value if detection.x < left_third else Instruction.MOVE_LEFT.value]
        return [Instruction.MOVE_RIGHT.value if detection.x < right_third else Instruction.MOVE_LEFT.value]

    @staticmethod
    def sky_is_horizontal_third(detection: Detection, tolerance=0.05):
        """Aligns the subject's head with the top horizontal third to leave room for the 'sky'."""
        y_top = detection.y - (detection.h / 2)
        target = 1 / 3
        if abs(y_top - target) > tolerance:
            return [Instruction.MOVE_DOWN.value if y_top > target else Instruction.MOVE_UP.value]
        return []

    @staticmethod
    def subject_feet_at_bottom(detection: Detection, tolerance=0.05):
        """Ensures the bottom of the bounding box is anchored to the bottom edge."""
        y_bottom = detection.y + (detection.h / 2)
        if y_bottom < (1.0 - tolerance):
            return [Instruction.MOVE_DOWN.value]
        return []

    def check_sloped_shoulders(self, detection, tolerance=0.08) -> List[str]:
        """Checks if the subject's shoulders are aligned horizontally."""
        # check if keypoints are detected
        if not detection.keypoints or len(detection.keypoints) < 7:
            return []

        l_shoulder, r_shoulder = detection.keypoints[5], detection.keypoints[6]

        # find the shoulder width and the vertical height difference between them
        y_diff = abs(l_shoulder[1] - r_shoulder[1])
        shoulder_width = abs(r_shoulder[0] - l_shoulder[0]) + 0.001  # +0.001 prevents division by zero

        # if the height difference is more than 8% of the shoulder width
        if (y_diff / shoulder_width) > tolerance:
            return [Instruction.LEVEL_SHOULDERS.value]

        return []

    def check_head_tilt(self, detection, max_angle_deg=8, neutral_zone=2) -> List[str]:
        """check head tilt to the side (confused dog)."""
        # check if keypoints are detected
        if not detection.keypoints or len(detection.keypoints) < 3:
            return []

        l_eye = detection.keypoints[1]
        r_eye = detection.keypoints[2]

        eye_dx = abs(r_eye[0] - l_eye[0])
        eye_dy = r_eye[1] - l_eye[1]
        if abs(eye_dx) < 1e-5:
            return []

        angle = math.degrees(math.atan2(eye_dy, eye_dx))

        # neutral band around straight head to avoid the instructions "flipping"
        if abs(angle) <= neutral_zone:
            return []
        if angle > max_angle_deg:
            return [Instruction.TILT_HEAD_LEFT.value]
        if angle < -max_angle_deg:
            return [Instruction.TILT_HEAD_RIGHT.value]

        return []

    def check_head_turn(self, detection, turn_tolerance=0.2) -> List[str]:
        """check if the head is turned do a certain direction"""
        # check if keypoints are detected
        if not detection.keypoints or len(detection.keypoints) < 7:
            return []

        nose = detection.keypoints[0]
        l_eye,r_eye = detection.keypoints[1], detection.keypoints[2]

        # calculate horizontal distance from nose to each eye, the distance between them and the difference
        l_eye_dist = abs(nose[0] - l_eye[0])
        r_eye_dist = abs(r_eye[0] - nose[0])
        eye_width = abs(r_eye[0] - l_eye[0]) + 0.001
        asymmetry = abs(l_eye_dist - r_eye_dist)

        # check the ratio relative to the tolerance
        if (asymmetry / eye_width) > turn_tolerance:
            if l_eye_dist > r_eye_dist:
                return [Instruction.TURN_HEAD_LEFT.value]
            else:
                return [Instruction.TURN_HEAD_RIGHT.value]

        return []

    def check_multi_angle_slouch(self, detection, hunch_tolerance=1.5) -> List[str]:
        """check for a forward slouch by comparing the neck length to the vertical size of the face."""
        # check if keypoints are detected
        if not detection.keypoints or len(detection.keypoints) < 7:
            return []

        nose = detection.keypoints[0]
        l_eye, r_eye = detection.keypoints[1], detection.keypoints[2]
        mid_eye_y = (l_eye[1] + r_eye[1]) / 2.0
        l_shoulder, r_shoulder = detection.keypoints[5], detection.keypoints[6]
        reference_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2.0

        # calculate "face scale": vertical distance from the eye down to the nose
        face_scale = abs(nose[1] - mid_eye_y)
        if face_scale < 0.001: # prevent division by zero if the face is tiny/glitched
            return []
        neck_length = reference_shoulder_y - nose[1] # calculate the neck length

        # check the ratio relative to the tolerance
        if (neck_length / face_scale) < hunch_tolerance:
            return [Instruction.STAND_STRAIGHT.value]

        return []

    def get_combined_detection(self, detections: List[Detection]) -> Optional[Detection]:
        """Tool to merge a couple/group into one detection for shared framing rules."""
        if not detections: return None
        if len(detections) == 1: return detections[0]

        x_left = min(d.x - d.w / 2 for d in detections)
        x_right = max(d.x + d.w / 2 for d in detections)
        y_top = min(d.y - d.h / 2 for d in detections)
        y_bottom = max(d.y + d.h / 2 for d in detections)

        w, h = x_right - x_left, y_bottom - y_top
        return Detection(x=x_left + w / 2, y=y_top + h / 2, w=w, h=h)

    def _get_detections(self, frame):
        if not self._model:
            return []

        if hasattr(self._model, 'predict') or "yolo" in str(type(self._model)).lower():
            results = self._model(frame, conf=0.5, verbose=False)
            return self._process_yolo(results)

        return []

    @staticmethod
    def _process_yolo(results):
        detections = []
        res = results[0]
        names = res.names

        for i, box in enumerate(res.boxes):
            xywhn = box.xywhn[0].tolist()
            cls_id = int(box.cls[0])
            label = names.get(cls_id, "unknown")

            kp = res.keypoints.xyn[i].tolist() if hasattr(res, 'keypoints') and res.keypoints is not None else None

            detections.append(Detection(
                *xywhn,
                confidence=box.conf[0].item(),
                label=label,
                keypoints=kp
            ))
        return detections

    def __repr__(self):
        attrs = ", ".join(f"{k.lstrip('_')}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"





