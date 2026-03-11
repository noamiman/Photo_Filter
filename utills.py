import os
import cv2
from datetime import datetime


def save_image(frame, folder="Data\images", filter_name=""):
    """
    Saves an image frame to a specified folder with a timestamped filename.

    :param frame: The image array to save.
    :param folder: Directory name where images will be stored.
    :param filter_name: Optional name of the filter used, appended to the folder path.
    :return: The filepath if successful, None otherwise.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    folder = os.path.join(folder, filter_name) if filter_name else folder
    filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(folder, filename)

    print(f"Saving photo to: {filepath}")

    success = cv2.imwrite(filepath, frame)

    if success:
        print(f"Photo saved successfully: {filepath}")
        return filepath

    print("Failed to save photo.")
    return None
