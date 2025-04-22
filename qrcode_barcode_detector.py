import cv2
import numpy as np
from pyzbar.pyzbar import decode
from google.colab.patches import cv2_imshow
from google.colab import files
import os

# Step 1: Upload the image file
uploaded = files.upload()

# Step 2: Check if the image exists and proceed
image_path = "input_image.jpeg"  # Correct the file name to match your uploaded file
print("Attempting to read image from:", image_path)

# Function to detect QR/barcodes
def detect_qr_barcode(image_path):
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: The image file {image_path} does not exist.")
        return

    # Read the input image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to read the image. Check the file path: {image_path}")
        return
    else:
        print("Image read successfully.")

    # Decode barcodes and QR codes
    detected_objects = decode(image)
    
    if not detected_objects:
        print("No QR codes or barcodes detected in the image.")
        return
    
    for obj in detected_objects:
        # Get the data and type (QR code or barcode)
        data = obj.data.decode('utf-8')
        code_type = obj.type
        
        # Get coordinates of the bounding box
        points = obj.polygon
        pts = [(point.x, point.y) for point in points]
        
        # Draw bounding box
        cv2.polylines(image, [cv2.convexHull(np.array(pts, dtype='int32'))], True, (0, 255, 0), 3)
        
        # Put the data text on the image
        x, y = pts[0]
        cv2.putText(image, f"{code_type}: {data}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        print(f"Detected {code_type}: {data}")
    
    # Display the image with bounding boxes
    cv2_imshow(image)

# Step 3: Run the function with the uploaded image path
#image_path = "input_image.jpg"  # Replace with your image path
detect_qr_barcode(image_path)
