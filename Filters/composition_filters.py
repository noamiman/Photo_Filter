from Filters.base_filter import BaseFilter
from Filters.base_filter import Instruction


class CenteredFilter(BaseFilter):
    @property
    def description(self):
        return "Centers the subject in the frame."

    def _calculate_feedback(self, frame, detections):
        """
        Analyzes detections to generate navigational instructions for the user.
        :param frame: the current frame from the camera.
        :param detections: a list of detected objects with normalized coordinates.
        :return: string instructions for the user.
        """
        # No detections
        if not detections:
            return Instruction.SEARCHING.value, False

        # save person detections.
        person = detections[0]
        x_center = person.x

        # the limit of the check.
        tolerance = 0.05

        # if the person is too left.
        if x_center < 0.5 - tolerance:
            return Instruction.MOVE_RIGHT.value, False

        # if the person is too right.
        elif x_center > 0.5 + tolerance:
            return Instruction.MOVE_LEFT.value, False

        # else we are ready.
        return Instruction.READY.value, True

class RuleOfThirdsFilter(BaseFilter):
    @property
    def description(self):
        return "Aligns the subject with the vertical third lines (left or right)."

    def apply(self, frame):
        pass

