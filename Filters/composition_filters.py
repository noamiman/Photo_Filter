from Filters.base_filter import BaseFilter
from Filters.base_filter import Instruction


class CenteredFilter(BaseFilter):
    @property
    def description(self):
        return "Centers the subject in the frame."

    def _calculate_feedback(self, frame, detections):
        if not detections:
            return Instruction.SEARCHING.value, False

        person = detections[0]
        x_center = person.x

        tolerance = 0.05
        if x_center < 0.5 - tolerance:
            return Instruction.MOVE_RIGHT.value, False
        elif x_center > 0.5 + tolerance:
            return Instruction.MOVE_LEFT.value, False

        return Instruction.READY.value, True

class RuleOfThirdsFilter(BaseFilter):
    @property
    def description(self):
        return "Aligns the subject with the vertical third lines (left or right)."

    def apply(self, frame):
        pass

