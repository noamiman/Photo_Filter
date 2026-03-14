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
    BODY_EDGES = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # ידיים וכתפיים
        (5, 11), (6, 12), (11, 12),  # טורסו (גוף מרכזי)
        (11, 13), (13, 15), (12, 14), (14, 16)  # רגליים
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
        if not self.target_keypoints: return
        h, w, _ = frame.shape

        # צבעים: ירוק זוהר כשיש התאמה, לבן נקי כשמחפשים
        main_color = (0, 255, 0) if is_ready else (255, 255, 255)

        # יצירת שכבת Overlay שחורה לגמרי - עליה נצייר את הנפח השקוף
        overlay = np.zeros_like(frame)

        # פונקציית עזר להמרת נקודות מנורמלות לפיקסלים
        def get_pt(idx):
            kp = self.target_keypoints[idx]
            return (int(kp[0] * w), int(kp[1] * h))

        # --- 1. ציור הנפח המלא על ה-Overlay (לשקיפות) ---
        # ראש
        head_center = get_pt(0)
        head_radius = int(h * 0.06)
        cv2.circle(overlay, head_center, head_radius, main_color, -1)  # עיגול מלא

        # טורסו (גוף מרכזי)
        torso_pts = np.array([get_pt(5), get_pt(6), get_pt(12), get_pt(11)], np.int32)
        cv2.fillPoly(overlay, [torso_pts], main_color)  # פוליגון מלא

        # גפיים (ידיים ורגליים) - נצייר אותן כקווים עבים מאוד כדי לתת נפח
        limb_thickness = int(w * 0.03)  # עובי מסיבי (למשל 20 פיקסלים ב-HD)
        for edge in self.BODY_EDGES:
            p1, p2 = get_pt(edge[0]), get_pt(edge[1])
            cv2.line(overlay, p1, p2, main_color, limb_thickness, cv2.LINE_AA)

        # --- 2. מיזוג השכבות (יצירת אפקט השקיפות החדה) ---
        alpha = 0.25  # רמת שקיפות נמוכה (25% צבע, 75% רקע) - זה נותן את מראה הזכוכית
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # --- 3. הוספת קו מתאר (Outline) דק לחדות ---
        # זה הופך את הצללית לחדה ואלגנטית מעל הנפח השקוף
        outline_thick = int(w * 0.005)  # עובי דק
        if outline_thick < 1: outline_thick = 1

        cv2.circle(frame, head_center, head_radius, main_color, outline_thick, cv2.LINE_AA)

        # טורסו כשרטוט
        p5, p6, p11, p12 = get_pt(5), get_pt(6), get_pt(11), get_pt(12)
        cv2.line(frame, p5, p6, main_color, outline_thick, cv2.LINE_AA)
        cv2.line(frame, p6, p12, main_color, outline_thick, cv2.LINE_AA)
        cv2.line(frame, p12, p11, main_color, outline_thick, cv2.LINE_AA)
        cv2.line(frame, p11, p5, main_color, outline_thick, cv2.LINE_AA)

        # גפיים כשרטוט
        for edge in self.BODY_EDGES:
            if edge in [(5, 6), (5, 11), (6, 12), (11, 12)]: continue
            p1, p2 = get_pt(edge[0]), get_pt(edge[1])
            cv2.line(frame, p1, p2, main_color, outline_thick, cv2.LINE_AA)


    def _pick_file_ui(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg")])
        root.update()
        root.after(200, root.destroy)
        return path