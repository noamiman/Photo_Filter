from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, List, Optional, Any
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
    READY = "Ready! Shoot!"
    SEARCHING = "Searching for person..."


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

    def sky_is_horizontal_third(self, detection: Detection, tolerance=0.05):
        """Aligns the subject's head with the top horizontal third to leave room for the 'sky'."""
        y_top = detection.y - (detection.h / 2)
        target = 1 / 3
        if abs(y_top - target) > tolerance:
            return [Instruction.MOVE_DOWN.value if y_top > target else Instruction.MOVE_UP.value]
        return []

    def subject_feet_at_bottom(self, detection: Detection, tolerance=0.05):
        """Ensures the bottom of the bounding box is anchored to the bottom edge."""
        y_bottom = detection.y + (detection.h / 2)
        if y_bottom < (1.0 - tolerance):
            return [Instruction.MOVE_DOWN.value]
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

        if hasattr(self._model, 'predict') or str(type(self._model)).lower().find('yolo') > -1:
            results = self._model(frame, classes=[0], conf=0.5, verbose=False)
            return self._process_yolo(results)

        return []

    @staticmethod
    def _process_yolo(results):
        """
        YOLO-Pose (COCO) Keypoints (0-16):
        ----------------------------------
        0: Nose | 1-2: Eyes | 3-4: Ears
        5-6: Shoulders | 7-8: Elbows | 9-10: Wrists
        11-12: Hips | 13-14: Knees | 15-16: Ankles

        Format: [[x, y, conf], ...] (Normalized 0.0-1.0)
        """
        detections = []
        res = results[0]
        for i, box in enumerate(res.boxes):
            if int(box.cls[0]) != 0: continue

            xywhn = box.xywhn[0].tolist()

            kp = res.keypoints.xyn[i].tolist() if hasattr(res, 'keypoints') and res.keypoints is not None else None
            detections.append(Detection(*xywhn, confidence=box.conf[0].item(), keypoints=kp))
        return detections


    def __repr__(self):
        attrs = ", ".join(f"{k.lstrip('_')}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"





