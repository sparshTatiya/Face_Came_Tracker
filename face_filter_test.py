import os
import cv2 
import cv2.data
import numpy as np 
RED = "\033[31;1m"
GREEN = "\033[32;1m"


def apply_overlay(frame, overlay, x, y):
    # Get the region of the interest from the frame
    h, w = overlay.shape[:2]

    # Boundary Checking for the overlay position
    if x < 0:
        overlay = overlay[:, -x:]
        w = overlay.shape[1]
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h = overlay.shape[0]
        y = 0
    if x + w > frame.shape[1]:
        overlay = overlay[:, :frame.shape[1] - x]
        w = overlay.shape[1]
    if y + h > frame.shape[0]:
        overlay = overlay[:frame.shape[0] - y, :]
        h = overlay.shape[0]

    # Ensure the overlay has an alpha channel
    if overlay.shape[2] < 4:
        bgr = overlay 
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        alpha = np.where(gray > 240, 0, 255).astype(np.uint8)
        overlay = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        overlay[:, :, 3] = alpha
    
    # Split channels 
    b, g, r, a = cv2.split(overlay)

    # Optional: blur alpha channel to smooth edges
    a = cv2.GaussianBlur(a, (5, 5), 0)
    alpha = a.astype(float) / 255.0
    alpha = np.dstack([alpha, alpha, alpha]) # Shape (h, w, 3)

    # Region of Interest (ROI) from the frame
    roi = frame[y:y+h, x:x+w].astype(float) 
    overlay_rgb = cv2.merge([b, g, r]).astype(float)

    # Blend the overlay with the ROI using alpha channel
    blended = alpha * overlay_rgb + (1 - alpha) * roi
    frame[y:y+h, x:x+w] = blended.astype(np.uint8)
    return frame



def main():
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml')

    # Load the overlay image (E.g Sunglasses)
    # Make sure Image has a transparency channel (alpha channel)
    overlay_path = "sunglasses2.png" 
    if not os.path.isfile(overlay_path):
        print(f"{RED}Error:Overlay Image not found at {overlay_path}")
        print("Please provide a valid path to an overlay with transparency.")
        return
    
    overlay_orig = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    if overlay_orig is None:
        print(f"{RED}Error:Failed to load overlay image.")
        return
    
    # Start the video capture from webcam 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"{RED}Error: Could not opened web-cam.")
        return
    
    print(f"Face{GREEN}Face filter started. Press `q` to quit.")

    while True:
        # Read the frame from the web-cam 
        ret, frame = cap.read()
        if not ret:
            print(f"{RED}Error:Failed to captue frame.")
            break 
        # Convert frame to greyscale for face detection 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame 
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        

        # Apply overlay to each detected face
        for (x, y, w, h) in faces:
            # Resize overlay to fit the face region 
            overlay_width = int(1.1 * w)
            overlay_height = int(overlay_width * overlay_orig.shape[0]/overlay_orig.shape[1])
            overlay = cv2.resize(overlay_orig, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)
            y_offset = y - int(0.25 * h)
            x_offset = x - int(0.05 * w)

            # Apply the overlay 
            frame = apply_overlay(frame, overlay, x_offset, y_offset)

            # For Debugging: Draw the rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Dislay the result 
        cv2.imshow("Face Filter", frame)

        # Break the loop if `q` is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release Resources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
