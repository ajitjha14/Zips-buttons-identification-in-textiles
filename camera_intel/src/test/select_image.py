import os
import re
import cv2

def natural_sort_key(s):
    # Sort helper function to extract numbers from filenames
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def select_and_keep_image():
    # Directories for saved images and detection results
    save_dir = "/home/amir/Desktop/test/Taken pictures"
    output_dir = "/home/amir/Desktop/test/Detection_results"
    
    # Get a list of all image files in the save directory
    images = [f for f in os.listdir(save_dir) if f.endswith('.jpg')]
    
    # Sort images using natural sort to ensure numeric ordering (e.g., img1, img2, img10)
    images.sort(key=natural_sort_key)

    # Check if there are any images to display
    if not images:
        print("No images found in the directory!")
        return

    # Initialize index to track the current image being displayed
    current_index = 0

    while True:
        # Display the current image in the detection folder
        detection_img_path = os.path.join(output_dir, images[current_index])
        
        # Check if the detection image exists in the output directory
        if os.path.exists(detection_img_path):
            img = cv2.imread(detection_img_path)

            # Show the image in a window with the image name
            cv2.imshow(f"Image {current_index + 1}/{len(images)}: {images[current_index]}", img)
        else:
            print(f"Detection result for '{images[current_index]}' not found. Skipping display.")

        # Wait for user input to navigate or select an image
        key = cv2.waitKey(0)

        if key == ord('d'):  # 'd' for next (forward)
            cv2.destroyAllWindows()
            current_index = (current_index + 1) % len(images)  # Loop to the beginning if at the end
        elif key == ord('a'):  # 'a' for previous (backward)
            cv2.destroyAllWindows()
            current_index = (current_index - 1) % len(images)  # Loop to the end if at the beginning
        elif key == ord('s'):  # 's' for select
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):  # 'q' for quit
            print("Selection process aborted.")
            return

    # Get the selected image file
    selected_image = images[current_index]
    selected_image_name = os.path.splitext(selected_image)[0]

    # Delete all other images and their corresponding text files and detection results
    for image in images:
        if image != selected_image:
            # Delete the image file from "Taken pictures" folder
            os.remove(os.path.join(save_dir, image))

            # Delete the corresponding txt file from "Detection_results" folder
            txt_file = os.path.join(output_dir, f"{os.path.splitext(image)[0]}.txt")
            if os.path.exists(txt_file):
                os.remove(txt_file)

            # Delete the corresponding image file from "Detection_results" folder
            detection_image = os.path.join(output_dir, image)
            if os.path.exists(detection_image):
                os.remove(detection_image)

    # Display confirmation
    print(f"Image '{selected_image}' and its result have been kept.")
    print("All other images and their result files have been deleted.")

if __name__ == "__main__":
    select_and_keep_image()

