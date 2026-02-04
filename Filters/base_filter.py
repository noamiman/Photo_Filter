from abc import ABC, abstractmethod
from enum import Enum

class Complexity(Enum):
    """
    Enum representing the complexity level of a filter.
    """
    LOW = 1
    MEDIUM = 2
    HIGH = 3

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
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                xywhn = box.xywhn[0].tolist()

                class SimpleBox: pass

                b = SimpleBox()
                b.x, b.y, b.w, b.h = xywhn
                detections.append(b)
        return detections

    def __repr__(self):
        attrs = ", ".join(f"{k.lstrip('_')}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"





