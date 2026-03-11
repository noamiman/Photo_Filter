import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from Filters.base_filter import BaseFilter, Instruction, Complexity


class UniversalTemplateFilter(BaseFilter):
    # connecting lines from points that are supposed to be linked
    # such as arm keypoints, leg key points - from given 16 keypoints
    EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    def __init__(self, name, model, template_image_path=None):
        super().__init__(name, Complexity.HIGH, model)
        self.target_keypoints = None
        self.target_box = None  # [x, y, w, h] מנורמל
        self._template_path = template_image_path

    @property
    def description(self):
        if self.target_keypoints:
            return "Fill the stencil: Adjust position and distance to match."
        return "Pick an image to create a fixed pose stencil."

    def _get_pose_data(self, frame):
        if not self._model: return None, None
        results = self._model(frame, conf=0.5, verbose=False)
        if results and len(results[0].keypoints) > 0:
            kp = results[0].keypoints.xyn[0].tolist()
            box = results[0].boxes.xywhn[0].tolist()
            return kp, box
        return None, None

    def _extract_template(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        kp, box = self._get_pose_data(img)
        if kp:
            self.target_keypoints = kp
            self.target_box = box
            return box
        return None

    def _calculate_feedback(self, frame, detections):
        curr_kp, curr_box = self._get_pose_data(frame)
        if not self.target_keypoints: return ["No template"], False
        if not curr_kp: return ["Searching for person..."], False

        feedback = []

        # --- בדיקת עומק (התקרבות/התרחקות) ---
        # אנחנו משווים את הרוחב והגובה של התיבה החוסמת (Bounding Box)
        target_w = self.target_box[2]
        curr_w = curr_box[2]

        depth_tol = 0.04
        if curr_w < target_w - depth_tol:
            feedback.append(Instruction.COME_CLOSER.value)
        elif curr_w > target_w + depth_tol:
            feedback.append(Instruction.STEP_BACK.value)

        # --- בדיקת מיקום (ימינה/שמאלה/למעלה/למטה) ---
        target_x, target_y = self.target_box[0], self.target_box[1]
        curr_x, curr_y = curr_box[0], curr_box[1]

        pos_tol = 0.03
        if curr_x < target_x - pos_tol:
            feedback.append(Instruction.MOVE_RIGHT.value)
        elif curr_x > target_x + pos_tol:
            feedback.append(Instruction.MOVE_LEFT.value)

        if curr_y < target_y - pos_tol:
            feedback.append(Instruction.MOVE_DOWN.value)
        elif curr_y > target_y + pos_tol:
            feedback.append(Instruction.MOVE_UP.value)

        # --- בדיקת פוזה ספציפית (דיוק איברים) ---
        # אם המיקום הכללי סביר, נבדוק נקודות קריטיות (מרפקים וברכיים)
        critical_joints = [7, 8, 13, 14]
        pose_errors = 0
        for idx in critical_joints:
            dist = np.linalg.norm(np.array(curr_kp[idx]) - np.array(self.target_keypoints[idx]))
            if dist > 0.08: pose_errors += 1

        if pose_errors > 1:
            feedback.append("Align your limbs to the stencil")

        if not feedback:
            return [Instruction.READY.value], True
        return feedback, False

    def apply(self, frame):
        if self.target_keypoints is None:
            if not self._template_path:
                self._template_path = self._pick_file_ui()
            if self._template_path:
                self._extract_template(self._template_path)
            if self.target_keypoints is None:
                return ["Please select an image"], False

        feedback, is_ready = self._calculate_feedback(frame, None)
        self._draw_fixed_stencil(frame, is_ready)
        return feedback, is_ready

    def _draw_fixed_stencil(self, frame, is_ready):
        """מצייר שבלונה קשיחה ומסיבית על הפריים"""
        if not self.target_keypoints: return
        h, w, _ = frame.shape

        # צבעים: ירוק זוהר כשיש התאמה, לבן חצי שקוף כשמחפשים
        color = (0, 255, 0) if is_ready else (255, 255, 255)
        overlay = frame.copy()

        # 1. ציור השלד על שכבת ה-Overlay (ליצירת אפקט חצי שקוף)
        for edge in self.EDGES:
            p1_idx, p2_idx = edge
            kp1, kp2 = self.target_keypoints[p1_idx], self.target_keypoints[p2_idx]
            if kp1[0] > 0 and kp2[0] > 0:
                pt1 = (int(kp1[0] * w), int(kp1[1] * h))
                pt2 = (int(kp2[0] * w), int(kp2[1] * h))
                # מצייר קו עבה "שבלוני"
                cv2.line(overlay, pt1, pt2, color, 8, cv2.LINE_AA)

        # 2. ציור המפרקים
        for kp in self.target_keypoints:
            if kp[0] > 0:
                cv2.circle(overlay, (int(kp[0] * w), int(kp[1] * h)), 10, color, -1)

        # 3. מיזוג השכבות (נותן אפקט של שבלונה על המסך)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 4. ציור קווי מתאר חדים מעל להדגשה
        for edge in self.EDGES:
            p1_idx, p2_idx = edge
            kp1, kp2 = self.target_keypoints[p1_idx], self.target_keypoints[p2_idx]
            if kp1[0] > 0 and kp2[0] > 0:
                pt1 = (int(kp1[0] * w), int(kp1[1] * h))
                pt2 = (int(kp2[0] * w), int(kp2[1] * h))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    def _pick_file_ui(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg")])
        root.destroy()
        return path