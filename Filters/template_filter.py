import cv2
from Filters.base_filter import BaseFilter, Instruction, Complexity, Detection


class UniversalTemplateFilter(BaseFilter):
    def __init__(self, name, model, template_image_path):
        # אנחנו קוראים ל-super בלי מודל בהתחלה, המנוע יזריק אותו ב-set_filter
        super().__init__(name, Complexity.HIGH, model)
        self.target_box = None
        self.object_name = "Object"

        # אם המודל כבר קיים (הועבר ב-init), ננתח את התמונה מיד
        if model:
            self.target_box = self._extract_template(template_image_path)

    def _get_detections(self, frame):
        """
        מזהה את כל סוגי האובייקטים שהמודל מכיר
        """
        if not self._model:
            return []

        # הרצה על כל המחלקות (בלי להגביל ל-classes=[0])
        results = self._model(frame, conf=0.5, verbose=False)
        detections = []

        if results and len(results[0].boxes) > 0:
            # שומרים גם את שם המחלקה של האובייקט הראשון שזוהה בתבנית
            class_id = int(results[0].boxes[0].cls[0])
            self.object_name = self._model.names[class_id]

            for box in results[0].boxes:
                xywhn = box.xywhn[0].tolist()
                detections.append(Detection(*xywhn))
        return detections

    def _extract_template(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error: Could not load image at {image_path}")
            return None

        print(f"🔍 Analyzing template: {image_path}...")
        template_dets = self._get_detections(img)

        if not template_dets:
            print("⚠️ No objects detected in the template image!")
            return None

        print(f"✅ Target locked on: {self.object_name}")
        return template_dets[0]

    @property
    def description(self):
        return f"Template: Match the {self.object_name} position"

    def _calculate_feedback(self, frame, detections):
        if not self.target_box:
            return "No target defined", False
        if not detections:
            return [f"Searching for {self.object_name}..."], False

        curr = detections[0]
        tgt = self.target_box

        feedback = []
        tol = 0.06  # רמת גמישות

        # השוואת מיקום וגודל
        if curr.x < tgt.x - tol:
            feedback.append(Instruction.MOVE_RIGHT.value)
        elif curr.x > tgt.x + tol:
            feedback.append(Instruction.MOVE_LEFT.value)

        if curr.y < tgt.y - tol:
            feedback.append(Instruction.MOVE_DOWN.value)
        elif curr.y > tgt.y + tol:
            feedback.append(Instruction.MOVE_UP.value)

        if curr.w < tgt.w - tol:
            feedback.append(Instruction.COME_CLOSER.value)
        elif curr.w > tgt.w + tol:
            feedback.append(Instruction.STEP_BACK.value)

        if not feedback:
            return [Instruction.READY.value], True
        return feedback, False

    def apply(self, frame):
        # אם עוד לא חילצנו תבנית (כי המודל הגיע באיחור מהמנוע)
        if self.target_box is None and self._model is not None:
            # כאן צריך להיזהר - עדיף לחלץ את התבנית ב-set_filter או בטעינה הראשונה
            pass

        feedback, is_ready = super().apply(frame)
        self._draw_guidelines(frame, is_ready)
        return feedback, is_ready

    def _draw_guidelines(self, frame, is_ready):
        if not self.target_box: return

        h, w, _ = frame.shape
        t = self.target_box

        x1, y1 = int((t.x - t.w / 2) * w), int((t.y - t.h / 2) * h)
        x2, y2 = int((t.x + t.w / 2) * w), int((t.y + t.h / 2) * h)

        color = (0, 255, 0) if is_ready else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"TARGET {self.object_name.upper()}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)