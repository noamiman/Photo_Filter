import os
import cv2
from datetime import datetime

def save_image(frame, folder="images"):
    """
    Saves an image frame to a specified folder with a timestamped filename.

    :param frame: The image array to save.
    :param folder: Directory name where images will be stored.
    :return: The filepath if successful, None otherwise.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"photo_{timestamp}.jpg"
    filepath = os.path.join(folder, filename)

    success = cv2.imwrite(filepath, frame)

    if success:
        print(f"Photo saved successfully: {filepath}")
        return filepath

    print("Failed to save photo.")
    return None