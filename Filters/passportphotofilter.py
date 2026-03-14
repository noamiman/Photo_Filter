from Filters.base_filter import *
import cv2


class PassportFilter(BaseFilter):

    def __init__(self, model=None):
        super().__init__(name="Passport Photo", complexity=Complexity.HIGH, model=model)

    @property
    def description(self):
        return "Strict composition for official ID and Passport photos."

    def _draw_template(self, frame, detection=None, ready=False):
        """
        Draws a professional, highly visual alignment UI.
        Features a static target silhouette and dynamic positioning arrows.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # --- MATHEMATICAL TARGETS ---
        target_center_x = w // 2
        target_eye_y = int(h * 0.38)

        # Calculate an "ideal" head size for the silhouette based on our 0.28-0.48 scale (aiming for 0.35)
        ideal_face_w = int(w * 0.35)
        ideal_face_h = int(ideal_face_w * 1.4)  # Standard human face ratio
        target_head_center_y = target_eye_y + int(ideal_face_h * 0.1)  # Center of head is slightly below eyes

        if ready:
            # ==========================================
            # SUCCESS STATE: BOLD GREEN
            # ==========================================
            color = (0, 255, 0)  # Bright Green
            thickness = 3

            # Draw a thick, solid oval confirming placement
            cv2.ellipse(overlay, (target_center_x, target_head_center_y),
                        (ideal_face_w // 2, ideal_face_h // 2), 0, 0, 360, color, thickness)

            # Draw a bold center crosshair
            cv2.line(overlay, (target_center_x - 30, target_eye_y), (target_center_x + 30, target_eye_y), color,
                     thickness)
            cv2.line(overlay, (target_center_x, target_eye_y - 20), (target_center_x, target_eye_y + 20), color,
                     thickness)

            # Optional: Add a subtle green vignette around the borders of the screen
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 100, 0), 10)

            alpha = 0.7  # Make the green pop heavily

        else:
            # ==========================================
            # GUIDANCE STATE: GRAY SILHOUETTE & ARROWS
            # ==========================================
            color = (200, 200, 200)  # Light Gray

            # 1. Draw the static target head oval (thin)
            cv2.ellipse(overlay, (target_center_x, target_head_center_y),
                        (ideal_face_w // 2, ideal_face_h // 2), 0, 0, 360, color, 1)

            # 2. Draw the exact mathematical "Safe Zone" Box
            # Based on dynamic tolerances: center=0.15, height=0.20 of ideal eye width
            ideal_eye_w = ideal_face_w // 2
            v_tol = int(ideal_eye_w * 0.20)
            h_tol = int(ideal_eye_w * 0.15)

            # Draw corner brackets for the safe zone
            cv2.rectangle(overlay, (target_center_x - h_tol, target_eye_y - v_tol),
                          (target_center_x + h_tol, target_eye_y + v_tol), color, 1)

            # Draw crosshairs reaching out to the edges
            cv2.line(overlay, (0, target_eye_y), (w, target_eye_y), color, 1)
            cv2.line(overlay, (target_center_x, 0), (target_center_x, h), color, 1)

            # 3. Dynamic Feedback: Show the user where they are vs where they should be
            if detection and detection.keypoints and len(detection.keypoints) >= 3:
                kp = detection.keypoints
                l_eye, r_eye = kp[1], kp[2]

                # Get the user's actual current center
                current_eye_y = int(((l_eye[1] + r_eye[1]) / 2) * h)
                current_center_x = int(((l_eye[0] + r_eye[0]) / 2) * w)

                # Draw a bright orange dot on the user's actual face center
                current_color = (0, 140, 255)  # Bright Orange (BGR)
                cv2.circle(overlay, (current_center_x, current_eye_y), 5, current_color, -1)

                # Draw a line connecting the user's face to the target box
                # Visually says: "Move this dot into the box!"
                cv2.line(overlay, (current_center_x, current_eye_y), (target_center_x, target_eye_y), current_color, 2)

            alpha = 0.4  # Keep the gray guides subtle so they don't distract

        # Apply the overlay to the actual frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


    def _check_face_size(self, detection: Detection, min_scale=0.28, max_scale=0.48) -> List[str]:
        """check if the face is the correct size for the photo."""
        if not detection.keypoints or len(detection.keypoints) < 5:
            return []

        l_ear, r_ear = detection.keypoints[3], detection.keypoints[4]
        face_width = abs(r_ear[0] - l_ear[0])

        if face_width < min_scale:
            return [Instruction.COME_CLOSER.value]

        elif face_width > max_scale:
            return [Instruction.STEP_BACK.value]

        return []

    def _check_eye_level(self, detection: Detection, target_y_pct=0.38, base_tolerance_pct=0.20) -> List[str]:
        """check if the eye level is at the correct coords (dynamically)."""
        if not detection.keypoints or len(detection.keypoints) < 7:
            return []

        l_eye, r_eye = detection.keypoints[1], detection.keypoints[2]
        eye_y_pct = (l_eye[1] + r_eye[1]) / 2.0
        eye_width = abs(r_eye[0] - l_eye[0])
        if eye_width < 1e-5:
            return []

        # calculate dynamic tolerance (+/- 20% of the subject's eye width)
        dynamic_tolerance = eye_width * base_tolerance_pct

        # check if the subject are too high or too low relative to the target line
        if eye_y_pct < (target_y_pct - dynamic_tolerance):
            return [Instruction.MOVE_DOWN.value]
        elif eye_y_pct > (target_y_pct + dynamic_tolerance):
            return [Instruction.MOVE_UP.value]

        return []

    def _check_strict_centering(self, detection: Detection, center_tolerance=0.15) -> List[str]:
        """check if the face is at the center of the frame."""
        if not detection.keypoints or len(detection.keypoints) < 3:
            return []

        l_eye, r_eye = detection.keypoints[1], detection.keypoints[2]
        face_center_x = (l_eye[0] + r_eye[0]) / 2.0 # calculate the true center of the face (bridge of the nose)

        if face_center_x < (0.5 - center_tolerance):
            return [Instruction.MOVE_RIGHT.value]
        elif face_center_x > (0.5 + center_tolerance):
            return [Instruction.MOVE_LEFT.value]

        return []

    @staticmethod
    def _check_head_pitch(detection: Detection, chin_up_tolerance=0.10, chin_down_tolerance=0.80) -> List[str]:
        """check if the head is facing too much up/down."""
        if not detection.keypoints or len(detection.keypoints) < 5:
            return []

        nose = detection.keypoints[0]
        l_eye, r_eye = detection.keypoints[1], detection.keypoints[2]
        l_ear, r_ear = detection.keypoints[3], detection.keypoints[4]

        mid_ear_y = (l_ear[1] + r_ear[1]) / 2.0 # calculate the midpoint between the ears
        eye_width = abs(r_eye[0] - l_eye[0])

        if eye_width < 0.001:  # prevent division by zero
            return []

        # the ratio: vertical distance from ears to nose
        pitch_ratio = (nose[1] - mid_ear_y) / eye_width

        if pitch_ratio < chin_up_tolerance:
            return [Instruction.CHIN_DOWN.value]
        if pitch_ratio > chin_down_tolerance:
            return [Instruction.CHIN_UP.value]

        return []

    def _calculate_feedback(self, frame, detections):
        self._draw_template(frame)

        if not detections:
            return [Instruction.SEARCHING.value], False

        main_person = detections[0]

        # 1. FUNDAMENTALS (Get them in the box)
        if err := self._check_face_size(main_person): return err, False

        # 2. POSITIONING (Center them)
        if err := self._check_strict_centering(main_person): return err, False
        if err := self._check_eye_level(main_person): return err, False

        # 3. MACRO POSTURE (Body alignment)
        if err := self.check_sloped_shoulders(main_person, tolerance=0.1): return err, False
        if err := self.check_multi_angle_slouch(main_person, hunch_tolerance=1.45): return err, False

        # 4. MICRO HEAD ALIGNMENT (Fine-tuning the face)
        if err := self.check_head_turn(main_person, turn_tolerance=0.15): return err, False
        if err := self._check_head_pitch(main_person): return err, False
        if err := self.check_head_tilt(main_person): return err, False

        # 5. SUCCESS
        self._draw_template(frame, main_person, ready=True)
        return [Instruction.READY.value], True