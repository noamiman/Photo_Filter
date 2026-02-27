from Filters.base_filter import BaseFilter, Instruction
import cv2
import numpy as np


class HeadroomFilter(BaseFilter):
    def __init__(self, name, complexity, model=None, traget_sky_ratio=0.3):
        super().__init__(name, complexity, model)
        self.target_sky_ratio = traget_sky_ratio
        self.margin = 0.05

    @property
    def description(self):
        return "Maintains the ideal gap between the head and the top edge."

    @staticmethod
    def _check_sky_color(frame, top_y):
        """
        Checks the color and texture of the area above the person's head.
        Differentiates between real sky (blue) and potential indoor walls (white).
        """
        height, width, _ = frame.shape
        head_pixel_y = int(top_y * height)

        # If the head is too close to the top, there's no space for sky
        if head_pixel_y <= 10:
            return 0

        # Slicing the region above the head
        sky_region = frame[0:head_pixel_y, :]
        total_pixels = sky_region.shape[0] * sky_region.shape[1]

        # Convert to HSV for color detection
        hsv_region = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)

        # Blue Mask (High confidence for sky)
        lower_blue = np.array([90, 50, 70])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv_region, lower_blue, upper_blue)
        blue_ratio = cv2.countNonZero(mask_blue) / total_pixels

        # 2. White/Bright Mask (Lower confidence - could be clouds OR ceiling)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        mask_white = cv2.inRange(hsv_region, lower_white, upper_white)
        white_ratio = cv2.countNonZero(mask_white) / total_pixels

        # Edge Detection (To detect indoor clutter/corners)
        # Sky is smooth, rooms have corners and edges.
        gray_region = cv2.cvtColor(sky_region, cv2.COLOR_BGR2GRAY)
        # Blur slightly to reduce digital noise
        blurred = cv2.GaussianBlur(gray_region, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edge_ratio = cv2.countNonZero(edges) / total_pixels

        # If the area is too "busy" (too many edges), it's likely not sky
        if edge_ratio > 0.05:
            return 0

        # If we have blue, we are much more confident it's sky.
        if blue_ratio > 0.1:
            # Combination of blue and white is great (blue sky with clouds)
            return blue_ratio + white_ratio
        else:
            # If it's only white, it's risky. We penalize it by 50%
            # to ensure it only hits READY if the area is huge and perfectly clear.
            return white_ratio * 0.5

    def _calculate_feedback(self, frame, detections):
        if not detections:
            return Instruction.SEARCHING.value, False

        person = detections[0]
        if person.keypoints and len(person.keypoints) > 0:
            top_y = person.keypoints[0][1]
        else:
            top_y = person.y - (person.h / 2)

        sky_confidence = self._check_sky_color(frame, top_y)

        if sky_confidence < 0.3:
            return Instruction.FIND_SKY.value, False

        if top_y < self.target_sky_ratio - self.margin:
            return Instruction.MOVE_DOWN.value, False

        if top_y > self.target_sky_ratio + self.margin:
            return Instruction.MOVE_UP.value, False

        return Instruction.READY.value, True


class DistanceFilter(BaseFilter):
    @property
    def description(self):
        return "Guides the photographer to the correct distance (Portrait vs Full Body)."

    def apply(self, frame):
        pass