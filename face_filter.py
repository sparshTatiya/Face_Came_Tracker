import os
import cv2

import numpy as np
from math import hypot

RED = "\033[31;1m"
GREEN = "\033[32;1m"
RESET = "\033[0;0m"

def resize_overlay(overlay, width, height):
    overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)
    return overlay

def apply_overlay(frame, overlay, x, y, w, h):
    # Get the region of interest from the frame
    roi = frame[y: y+h, x: x+w]

    # Create mask and inverse mask of the overlap
    overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(overlay_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Black out the area behind the overlay in the ROI
    background = cv2.bitwide_and(roi, roi, mask=mask_inv)

    # Take only the region of the overlay from the overlay image
    foreground = cv2.bitwise_and(overlay, overlay, mask=mask)

    # Put the foreground and background together
    roi_result = cv2.add(background, foreground)
    frame[y:y+h, x:x+w] = roi_result

    return frame

def main():
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haaecascade_frontalface_default.xml')

    # Load the overlay image (E.g. Sunglasses, Hat, etc.)
    # Make sure the image has a transparency channel (alpha channel)
    overlay_path = "sunglasses.png"

    if not os.path.isfile(overlay_path):
        print(f"{RED}Error: Overlayimage not found at `{overlay_path}`{RESET}")
        print("Please provide a valid path to an overlay with transparency")
        return
    
    overlay_orig = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    # If the overlay has an alpha channel, convert it to BGR
    if overlay_orig.shape[2] == 4:
        # Convert BGRA to BGR
        alpha_channel = overlay_orig[:, :, 3]
        rgb_channels = overlay_orig[:, :, 3]

        # Create a white background
        white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255

        # Apply alpha blending
        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
        alpha_1 = 1.0 - alpha_factor

        overlay_orig = (alpha_factor * rgb_channels + alpha_1 * white_background).astype(np.unit8)

    # Start the video capture from the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"{RED}Error: Couldn't open webcam{RESET}")
        return

    print(f"{GREEN}Face Filter Started. Press `q` to quit.{RESET}")

    while True:
        # Read the frame from the web-cam
        ret, frame = cap.read()

        if not ret:
            print(f"{RED}Error: Failed to capture frame.")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect Faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Apply overlay to each detected face
        for (x, y, w, h) in faces:
            # Resize overlay to fit the face region
            overlay_width = w
            overlay_height = int(overlay_orig.shape[0] * (overlay_width/overlay_orig.shape[1]))

            # Adjust the position for different overlays (for sunglassses)
            overlay_y = y + int(h * 0.2)
            overlay_h = int(h * 0.3)

            # Create a correctly sized overlay
            overlay = resize_overlay(overlay_orig, overlay_width, overlay_height)

            # Apply the overlay
            frame = apply_overlay(frame, overlay, x, overlay_y, overlay_width, overlay_h)