from ultralytics import YOLO
import os
import glob

# Load the YOLO model
model = YOLO("/home/amir/Desktop/test/best.pt")  # Path to your trained YOLOv8 model

# Set the input folder containing the images and output folder
input_folder = "/home/amir/Desktop/test"  # Folder containing the images
output_folder = "/home/amir/Desktop/test/Detection_results"  # Path to save the detection results

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all image files in the input folder (you can modify the extensions as needed)
image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))  # Adjust extension if needed, e.g., .png, .jpeg

# Run inference on each image in the folder
for image_path in image_paths:
    print(f"Processing {image_path}...")
    
    # Run inference
    results = model.predict(image_path)

    # Save each result manually
    for idx, result in enumerate(results):
        # Generate output filename and path
        output_path = os.path.join(output_folder, f"detected_{os.path.basename(image_path)}")
        
        # Plot the results and save to the specified output folder
        result.plot(save=True, filename=output_path)

print(f"All detection results saved to {output_folder}")

