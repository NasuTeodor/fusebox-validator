import cv2
import pickle

# ROI selection globals
roi_coordinates = []
drawing = False
ix, iy = -1, -1

def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, roi_coordinates
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        roi_coordinates = [(x, y)]
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = frame.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Configure ROI', img_copy)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_coordinates.append((x, y))
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)

# Initialize camera with specified settings
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Verify camera initialization
if not cap.isOpened():
    print("Error: Could not open camera at index 2")
    print("Possible fixes:")
    print("1. Check camera connection")
    print("2. Try different index (0, 1, or 2)")
    print("3. Verify DirectShow compatibility")
    exit()

# Verify resolution
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {actual_width}x{actual_height}")

cv2.namedWindow('Configure ROI')
cv2.setMouseCallback('Configure ROI', draw_roi)

print("ROI Selection Guide:")
print("1. Click and drag to draw rectangle")
print("2. Press 's' to save ROI")
print("3. Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Display instructions
    cv2.putText(frame, "Drag ROI & Press 's' to Save", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show existing ROI if defined
    if len(roi_coordinates) == 2:
        cv2.rectangle(frame, roi_coordinates[0], roi_coordinates[1], (0, 255, 0), 2)
        
    cv2.imshow('Configure ROI', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        if len(roi_coordinates) == 2:
            # Normalize coordinates (x, y, w, h)
            x = min(roi_coordinates[0][0], roi_coordinates[1][0])
            y = min(roi_coordinates[0][1], roi_coordinates[1][1])
            w = abs(roi_coordinates[0][0] - roi_coordinates[1][0])
            h = abs(roi_coordinates[0][1] - roi_coordinates[1][1])
            
            # Save to pickle
            with open('roi_settings.pkl', 'wb') as f:
                pickle.dump({
                    'camera_index': 2,
                    'resolution': (1280, 720),
                    'roi': (x, y, w, h)
                }, f)
            print(f"Saved ROI: X={x}, Y={y}, W={w}, H={h}")
        break
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()