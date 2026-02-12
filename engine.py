import cv2
from ultralytics import YOLO
import torch
import os
from datetime import datetime


class CameraEngine:
    def __init__(self, model_path):
        # load the model into memory
        self.model = self._load_model(model_path)
        # initialize with no filter selected
        self.current_filter = None



    def _load_model(self, path):
        """
        load the model from a given path
        """
        try:
            print(f"Attempting to load model from: {path}")
            model = YOLO(path)

            # chose whether to use GPU or CPU based on availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            print(f"Model loaded successfully on {device.upper()}.")
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None

    def set_filter(self, filter_instance):
        """
        set the current filter and link it to the model
        """
        self.current_filter = filter_instance
        self.current_filter._model = self.model

    def run_live_camera(self):
        # live camera feed using OpenCV
        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Camera started. Press 'q' to quit.")

        while True:
            # read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                break

            # save a clean copy of the frame before drawing overlays
            clean_frame = frame.copy()

            # process the frame through the current filter
            feedback, is_ready = self.process_frame(frame)

            # visualize the feedback on the frame
            self._render_overlay(frame, feedback, is_ready)

            # show the frame with feedback
            cv2.imshow('Smart Camera Feedback', frame)

            key = cv2.waitKey(1) & 0xFF
            # exit on 'q' key press
            if key & 0xFF == ord('q'):
                break
            # Save the picture
            elif key == ord('w'):
                # We only want to allow saving if the filter says we are ready
                if is_ready:
                    self._save_image(clean_frame)
                else:
                    print("Not ready yet! Align the subject first.")

        # release the camera and close windows
        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        if self.current_filter is None:
            return "No filter selected", False
        return self.current_filter.apply(frame)

    def _render_overlay(self, frame, feedback, is_ready):
        """
        draws feedback text and visual cues on the frame
        """
        color = (0, 255, 0) if is_ready else (0, 255, 255)

        # write the feedback text on the frame
        cv2.putText(frame, feedback, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        filter_description = self.current_filter.description
        cv2.putText(frame, filter_description, (1100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        # draw a rectangle around the frame if ready
        if is_ready:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10)
            shoot_msg = "click on 'w' to take a picture"
            cv2.putText(frame, shoot_msg, (400, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    def _save_image(self, frame):
        # 1. Create a folder for photos if it doesn't exist
        folder = "images"
        if not os.path.exists(folder):
            os.makedirs(folder)

        # 2. Generate a unique filename using the current time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)

        # 3. Save the frame
        # Note: Use a copy of the frame BEFORE drawing the text/rectangles
        # if you want a clean photo!
        success = cv2.imwrite(filepath, frame)

        if success:
            print(f"Photo saved successfully: {filepath}")
        else:
            print("Failed to save photo.")



