from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple

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

class Instruction(Enum):
    """
    Enum representing various instructions for user guidance.
    """
    MOVE_LEFT = "← Move Left"
    MOVE_RIGHT = "Move Right →"
    MOVE_UP = "Tilt Up ↑"
    MOVE_DOWN = "Tilt Down ↓"
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

    def _get_detections(self, frame):
        """
        Extracts detections from the frame using the YOLO model.
        """
        if not self._model:
            return []

        # check results from the frame, restrict to class 0 (person) with confidence threshold 0.5
        results = self._model(frame, classes=[0], conf=0.5, verbose=False)
        detections = []

        # Extract person detections
        # x: The X-coordinate of the center of the bounding box.
        # y: The Y-coordinate of the center of the bounding box.
        # w: The width of the bounding box.
        # h: The height of the bounding box.
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                xywhn = box.xywhn[0].tolist()

                det = Detection(*xywhn)
                detections.append(det)

        return detections

    def __repr__(self):
        attrs = ", ".join(f"{k.lstrip('_')}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"





