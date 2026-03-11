from Filters.base_filter import BaseFilter
from Filters.base_filter import Instruction


class HorizonLevelerFilter(BaseFilter):
    @property
    def description(self):
        return "Ensures the camera is level and the horizon is not tilted."

    def apply(self, frame):
        pass


class SymmetryFilter(BaseFilter):
    @property
    def description(self):
        return "Helps achieve visual balance between the subject and the background."

    def apply(self, frame):
        pass


class HeroShotFilter(BaseFilter):
    @property
    def description(self):
        return "Hero Shot: Subject on a third, feet at bottom, and room for background/sky."

    def _calculate_feedback(self, frame, detections):
        # Handle no detections
        if not detections:
            return [Instruction.SEARCHING.value], False

        # Identify subject
        subject = self.get_combined_detection(detections)

        # Collect errors
        errors = []
        errors.extend(self.subject_on_third(subject))  # Vertical Thirds
        errors.extend(self.subject_feet_at_bottom(subject))  # Grounding
        errors.extend(self.sky_is_horizontal_third(subject))  # Headroom/Sky

        # If list is empty, composition is perfect
        if not errors:
            return [Instruction.READY.value], True

        return errors, False