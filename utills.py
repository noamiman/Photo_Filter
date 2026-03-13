import os
import cv2
from datetime import datetime


def save_image(frame, folder="Data/images", filter_name=""):
    """
    Saves an image frame to a specified folder with a timestamped filename.
    """
    # build the target folder path, optionally including the filter name as a subfolder
    target_folder = os.path.join(folder, filter_name) if filter_name else folder

    # make sure the target folder exists, if not, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created directory: {target_folder}")

    # build the filename with a timestamp to ensure uniqueness
    filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(target_folder, filename)

    print(f"Saving photo to: {filepath}")

    # save the image using OpenCV, and check if the operation was successful
    success = cv2.imwrite(filepath, frame)

    if success:
        print(f"Photo saved successfully: {filepath}")
        return filepath

    print("Failed to save photo. Check if the frame is valid and permissions are set.")
    return None
