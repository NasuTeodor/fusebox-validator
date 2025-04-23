import cv2
import numpy as np
import pickle as pkl

def update_mask_intensity(val):
    global mask_intensity
    mask_intensity = val

def main():
    global mask_intensity
    global rectangles
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Open webcam (2 for specific camera)
    try:
        with open("rectangles.pkl", "rb") as f:
            rectangles = pkl.load(f)
        print(f"Loaded {len(rectangles)} rectangles.\n{rectangles}")
    except FileNotFoundError:
        print("No previous rectangles found. Starting with an empty list.")
    exit()

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    width = 640  # Set desired width
    height = int((9 / 16) * width)  # Maintain 16:9 aspect ratio
    
    mask_intensity = 135  # Variable to control mask intensity (0 for black, 255 for white)
    
    cv2.namedWindow("Resized Video")
    cv2.createTrackbar("Mask Intensity", "Resized Video", 0, 255, update_mask_intensity)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        resized_frame = cv2.resize(frame, (width, height))
        
        # Create a variable-intensity mask of the same size
        mask = np.full_like(resized_frame, mask_intensity, dtype=np.uint8)
        
        # Apply the mask
        masked_frame = cv2.bitwise_and(resized_frame, mask)
        
        # Convert to grayscale
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Detect blobs using thresholding
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw rectangles around detected blobs
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Resized Video", resized_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
