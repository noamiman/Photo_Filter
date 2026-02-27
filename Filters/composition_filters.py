from Photo_Filter.Filters.base_filter import BaseFilter, Complexity
from Photo_Filter.Filters.base_filter import Instruction


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
        center = 0.5

        # if the person is too left.
        if x_center < center - tolerance:
            return Instruction.MOVE_RIGHT.value, False

        # if the person is too right.
        elif x_center > center + tolerance:
            return Instruction.MOVE_LEFT.value, False

        # else we are ready.
        return Instruction.READY.value, True


class RuleOfThirdsFilter(BaseFilter):
    @property
    def description(self):
        return "Aligns the subject with the vertical third lines (left or right)."

    def _calculate_feedback(self, frame, detections):
        if not detections:
            return [Instruction.SEARCHING.value], False

        # the feedbacks for the user
        curr_errors = []

        # get the detections
        person = detections[0]
        x_center, y_center, h = person.x, person.y, person.h

        target_x_left, target_x_right, target_y_top = 1 / 3, 2 / 3, 1 / 3
        tolerance = 0.1
        y_head = y_center - (h / 2)

        # check where is the subject
        is_at_left = abs(x_center - target_x_left) <= tolerance
        is_at_right = abs(x_center - target_x_right) <= tolerance

        # check logic, if something not working add to curr errors.
        if not (is_at_left or is_at_right):
            if x_center < target_x_left:
                curr_errors.append(Instruction.MOVE_RIGHT.value)
            elif x_center > target_x_right:
                curr_errors.append(Instruction.MOVE_LEFT.value)
            else:
                if abs(x_center - target_x_left) < abs(x_center - target_x_right):
                    curr_errors.append(Instruction.MOVE_LEFT.value)
                else:
                    curr_errors.append(Instruction.MOVE_RIGHT.value)

        is_y_ready = abs(y_head - target_y_top) <= tolerance
        if not is_y_ready:
            if y_head < target_y_top:
                curr_errors.append(Instruction.MOVE_DOWN.value)
            else:
                curr_errors.append(Instruction.MOVE_UP.value)

        # don't have errors, ready feedback
        if not curr_errors:
            return [Instruction.READY.value], True

        # there is errors, send to the render overlay function in the engine.
        return curr_errors, False


class LookRoomFilter(BaseFilter):
    def __init__(self, model=None):
        super().__init__(name="Look Room Filter", complexity=Complexity.MEDIUM, model=model)

    @property
    def description(self):
        return "Leaves empty space in the direction the subject is looking."

    def _calculate_feedback(self, frame, detections):
        if not detections:
            return Instruction.SEARCHING.value, False

        person = detections[0]
        if not person.keypoints or len(person.keypoints) < 5:
            return self._fallback_center(person.x)

        nose_x = person.keypoints[0][0]
        left_ear_x = person.keypoints[3][0]
        right_ear_x = person.keypoints[4][0]

        dist_to_left_ear = abs(nose_x - left_ear_x)
        dist_to_right_ear = abs(nose_x - right_ear_x)

        # make sure there won't be division by zero in the future calculations
        dist_to_left_ear = max(dist_to_left_ear, 0.0001)
        dist_to_right_ear = max(dist_to_right_ear, 0.0001)

        # calculate the gaze ratio
        target_x = 0.5
        gaze_threshold = 1.5

        # if the subject is looking significantly to the right, we want to leave more space on the right side
        if dist_to_right_ear / dist_to_left_ear > gaze_threshold:
            target_x = 1 / 3
        if dist_to_left_ear / dist_to_right_ear > gaze_threshold:
            target_x = 2 / 3
        tolerance = 0.05

        if person.x < target_x - tolerance:
            return Instruction.MOVE_RIGHT.value, False
        elif person.x > target_x + tolerance:
            return Instruction.MOVE_LEFT.value, False

        return Instruction.READY.value, True

    @staticmethod
    def _fallback_center(x_center):
        tolerance = 0.05
        if x_center < 0.5 - tolerance:
            return Instruction.MOVE_RIGHT.value, False
        elif x_center > 0.5 + tolerance:
            return Instruction.MOVE_LEFT.value, False
        return Instruction.READY.value, True

