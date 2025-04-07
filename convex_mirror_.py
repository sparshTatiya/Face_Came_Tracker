import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple 

RED = "\033[31;1m"
BLUE = "\033[32;1m"
RESET = "\033[0;0m"

def apply_mirror_effect(image:np.ndarray, effect_type:str="convex", strength:float=0.5, center:Tuple|None = None):
    # Copy the Input Image
    result = image.copy()

    # Getting the Height and Width of the Image
    height, width = image.shape[:2]
   
    # Use the Image center if not specified
    if center is None:
        center = (width//2, height//2) 

    # Create coordinate maps
    y_map, x_map = np.indices((height, width), dtype=np.float32)

    # Calculate distance from each pixel to the center 
    x_map = x_map - center[0]
    y_map = y_map - center[1]

    # Calculate normalized radial distance
    r = np.sqrt(x_map**2 + y_map ** 2)
    max_r = np.sqrt(width**2 + height**2)/2 
    normalized_r = r / max_r

    # Apply different formulas for convex or concave effect 
    if effect_type == "convex":
        # Concex Effect (bulging outwards)
        r_new = normalized_r * (1 - strength * (1 - normalized_r))
    else: 
        # Concave Effect (caving inward)
        r_new = normalized_r * (1 + strength * normalized_r)

    # Scale Back 
    r_new = r_new * max_r

    # Calculate the angle
    angle = np.arctan2(y_map, x_map)

    # Convert angular (polar) to cartensian coordiantes
    x_new = r_new * np.cos(angle) + center[0]
    y_new = r_new * np.sin(angle) + center[1]

    # Remap into Pixels 
    map_x = x_new.astype(np.float32)
    map_y = y_new.astype(np.float32)

    # Apply remapping
    result = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return result 

def interactive_mirror_effect(image_path, window_size=(800, 600)):
    
    image = cv2.imread(image_path)

    if image is None:
        print(f"{RED}Error: Could not load Image.{RESET}")
        return
    
    # Convert to RGB for consistent processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create Window
    window_width, window_height = window_size
    window_name = "Mirror Effect"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)

    # Default Parameters 
    effect_type = "convex"
    strength = 50 # 0-100 for trackbar 

    # Callback function for effect type 
    def on_effect_change(val):
        nonlocal effect_type 
        effect_type = "convex" if val == 0 else "concave"
        update_image()

    # Callback function for strength 
    def on_strength_change(val):
        nonlocal strength
        strength = val
        update_image()

    # Function to update image
    def update_image():
        # Apply Effect 
        result = apply_mirror_effect(image_rgb, effect_type=effect_type, strength=strength/100.0)

        # Convert back to BGR for display with OpenCV
        cv2.imshow("Mirror Effect", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    # Create trackbars
    cv2.createTrackbar("Effect (0=Convex, 1=Concave)", "Mirror Effect", 0, 1, on_effect_change)
    cv2.createTrackbar("Strength", "Mirror Effect", strength, 100, on_strength_change)

    # Intial Display
    update_image()

    # Wait for the key-press
    print(f"{RED}Press `ESC` to exit, `s` to save the current result{RESET}")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == 27: # Escape Key 
            break
        elif key == ord("s"):
            # Save the current result
            result = apply_mirror_effect(image_rgb, effect_type=effect_type, strength=strength/100.0)
            filename = f"{effect_type}_strength_{strength}.jpg"
            cv2.imwrite(filename, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"{BLUE}Saved as {filename}.")

    cv2.destroyAllWindows()


def sample_example(img_path:str):
    # Load an Image 
    image = cv2.imread(img_path)

    # Check if image is loaded
    if image is None:
        print("Error: Could'nt load the Image.")
        return
    
    # Convert Image from BGR to RGB (for display in matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply convex mirror effect 
    convex_image = apply_mirror_effect(image_rgb, effect_type="convex", strength=1)

    # Apply concave mirror effect 
    concave_image = apply_mirror_effect(image_rgb, effect_type="concave", strength=1)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Convex mirror Effect")
    plt.imshow(convex_image)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Concave mirror Effect")
    plt.imshow(concave_image)
    plt.axis("off")

    # Save Results 
    cv2.imwrite("MirrorImages/convex_result.jpg", cv2.cvtColor(convex_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("MirrorImages/concave_result.jpg", cv2.cvtColor(concave_image, cv2.COLOR_RGB2BGR))

    print("\033[32;1mResults saved as `convex_result.jpg` and `concave_result.jpg`") 

if __name__ == "__main__":
    #sample_example(os.path.join("ElonMusk.png"))

    interactive_mirror_effect("elon.png")