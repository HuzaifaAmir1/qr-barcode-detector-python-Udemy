import cv2
from pyzbar.pyzbar import decode
 
def detect_qr_barcode(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image. Check the file path.")
        return
 
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
    cv2.imshow("Detected QR/Barcodes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
# Example usage
image_path = "input_image.jpg"  # Replace with your image path
detect_qr_barcode(image_path)