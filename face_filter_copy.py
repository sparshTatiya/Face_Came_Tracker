import os
import cv2
import traceback

import numpy as np
from math import hypot

RED = "\033[31;1m"
GREEN = "\033[32;1m"
RESET = "\033[0;0m"

def resize_overlay(overlay, width, height):
    overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)
    return overlay

def apply_overlay(frame, overlay, x, y, w, h):
    # Make sure conditions are valid
    frame_h, frame_w = frame.shape[:2]

    # Adjust coordinates if they are out of bounds
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > frame_w:
        w = frame_w - x
    if y + h > frame_h:
        h = frame_h - y

    # Skip if any dimension is invalid
    if w <= 0 or h <= 0:
        return frame

    # Get the region of interest from the frame
    roi = frame[y: y+h, x: x+w]

    # Resize overlay to match ROI dimensions correctly
    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)


    """
    # Check if ROI and overlay have comppatible dimensions
    if roi.shape[0:2] != overlay.shape[0:2]:
        # Resize the overlay to match ROI dimension
        overlay = cv2.resize(overlay, (roi.shape[1], roi.shape[0]))

    # If overlay has 3 channels (RGB) but no alpha channels
    if overlay.shape[2] == 3:
        # Create mask and inverse mask of the overlap
        overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(overlay_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Ensure mask dimensions match ROI dimensions
        if mask.shape[0:2] != roi.shape[0:2]:
            mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
            mask_inv = cv2.resize(mask_inv, (roi.shape[1], roi.shape[0]))

        # Black out the area behind the overlay in the ROI
        background = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Take only the region of the overlay from the overlay image
        foreground = cv2.bitwise_and(overlay, overlay, mask=mask)

        # Put the foreground and background together
        roi_result = cv2.add(background, foreground)
    """

    # If overlay has 4 channels (BGRA) with alpha channels
    if overlay_resized.shape[2] == 4:
        # Extract RGB and alpha channels
        overlay_rgb = overlay[:, :, 0:3]
        alpha = overlay[:, :, 3]/255.0

        # Convert alpha channel to 3D (for broadcast)
        alpha = alpha[:, :, np.newaxis]

        # Blend overlay with ROI using alpha channel
        blended = cv2.multiply(1.0 - alpha, roi, dtype=cv2.CV_8U) + cv2.multiply(alpha, overlay_rgb, dtype=cv2.CV_8U)

        roi_result = blended.astype(np.uint8)

    else:
        # Create mask and inverse mask of the overlap
        overlay_gray = cv2.cvtColor(overlay_resized, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(overlay_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Black out the area behind the overlay in the ROI
        background = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Take only the region of the overlay from the overlay image
        foreground = cv2.bitwise_and(overlay, overlay, mask=mask)

        # Put the foreground and background together
        roi_result = cv2.add(background, foreground)

    frame[y:y+h, x:x+w] = roi_result
    return frame

def main():
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the overlay image (E.g. Sunglasses, Hat, etc.)
    # Make sure the image has a transparency channel (alpha channel)
    overlay_path = "sunglasses2.png"

    if not os.path.isfile(overlay_path):
        print(f"{RED}Error: Overlayimage not found at `{overlay_path}`{RESET}")
        print("Please provide a valid path to an overlay with transparency")
        return
    
    overlay_orig = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    # If the image doesn't have an alpha channel, we need to handle it differently
    if overlay_orig is None:
        print(f"{RED}Error: Could not load image {overlay_path}")
        return
    
    print(f"{GREEN}Loaded image with shape: {overlay_orig.shape}")

    # If the image doesn't have an alpha channel, we need to handle it differently
    if len(overlay_orig.shape) < 3 or overlay_orig.shape[2] != 4:
        print(f"{RED}Warning: Image doesn't have an alpha channel. Transparency won't work properly.")
        # Convert to BGR if it's grayscale
        if len(overlay_orig.shape) < 3:
            overlay_orig = cv2.cvtColor(overlay_orig, cv2.COLOR_GRAY2BGR)
    
    else:
        print(f"{GREEN}Alpha channel detected - transparency will be preserved")

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
            try:
                # Resize overlay to fit the face region
                overlay_width = w

                # Calculate height while maintaining the aspect ratio
                if overlay_orig.shape[1] > 0:
                    overlay_height = int(overlay_orig.shape[0] * (overlay_width/overlay_orig.shape[1]))
                else:
                    overlay_height = h // 3

                # Adjust the position for different overlays (for sunglassses)
                overlay_y = y + int(h * 0.2)
                overlay_x = x

                # Create a correctly sized overlay
                overlay = resize_overlay(overlay_orig, overlay_width, overlay_height)

                # Apply the overlay
                frame = apply_overlay(frame, overlay, overlay_x, overlay_y, overlay_width, overlay_height)

            except Exception as e:
                print(f"{RED}Error applying overlay: {e}")
                traceback.print_exc()
                # For debugging: Draw the rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)


        # Display the result
        cv2.imshow("Face Filter", frame)

        # Break the loop if `q` is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
