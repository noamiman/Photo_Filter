from Filters.base_filter import BaseFilter, Complexity, Instruction, Detection
from typing import List

class KitchenOutdoorFilter(BaseFilter):
    def __init__(self, model):
        # initialize the base filter with a name, complexity level, and the model
        super().__init__(name="Kitchen & Outdoor", complexity=Complexity.MEDIUM, model=model)

        # if has attribute set_classes, set the classes we want to detect for this filter
        if hasattr(self._model, 'set_classes'):
            # example of free classes for this filter
            self._model.set_classes(["refrigerator", "bird", "car"])

    @property
    def description(self):
        return "Detects refrigerators, birds, and cars to help with framing."

    def _calculate_feedback(self, frame, detections: List[Detection]):
        # make list of detections for each class
        fridges = [d for d in detections if d.label == "refrigerator"]
        birds = [d for d in detections if d.label == "bird"]
        cars = [d for d in detections if d.label == "car"]

        feedback = []

        # logic for feedback based on detected objects
        if fridges:
            # for example, if a fridge is detected, we want it to be centered in the frame
            feedback.extend(self.subject_centered(fridges[0]))

        if birds:
            # for example, if a bird is detected, we want it to be on one of the vertical thirds
            feedback.extend(self.subject_on_third(birds[0]))

        if not (fridges or birds or cars):
            return ["Searching for a fridge, bird, or car..."]

        return feedback if feedback else [Instruction.READY.value]