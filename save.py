import cv2
import pickle

# Global variables
rectangles = {i: [] for i in range(1, 9)}
current_type = 1
start_x, start_y = -1, -1
drawing = False
current_rectangle = None

COLOR_PALETTE = {
    1: (255, 0, 0),    # Blue
    2: (0, 255, 0),    # Green
    3: (0, 0, 255),    # Red
    4: (255, 255, 0),  # Cyan
    5: (255, 0, 255),  # Magenta
    6: (0, 255, 255),  # Yellow
    7: (128, 0, 255),  # Purple
    8: (128, 255, 128) # Light Green
}

# Initialize video capture with higher resolution
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set camera resolution to 1280x720 (HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Verify actual resolution
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution set to: {int(actual_width)}x{int(actual_height)}")

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

def mouse_callback(event, x, y, flags, param):
    global start_x, start_y, drawing, current_rectangle
    
    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        drawing = True
        current_rectangle = (start_x, start_y, x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_rectangle = (start_x, start_y, x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        sorted_x = sorted([start_x, current_rectangle[2]])
        sorted_y = sorted([start_y, current_rectangle[3]])
        final_rect = (sorted_x[0], sorted_y[0], sorted_x[1], sorted_y[1])
        rectangles[current_type].append(final_rect)
        current_rectangle = None

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", 1280, 720)
cv2.setMouseCallback("Video", mouse_callback)

print("Instructions:")
print("- 1-8: Fuse type | Draw: Click+drag | Save: S | Quit: Q")

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Draw all saved rectangles
    for fuse_type in rectangles:
        for rect in rectangles[fuse_type]:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), COLOR_PALETTE[fuse_type], 2)
    
    # Draw current rectangle
    if current_rectangle:
        cv2.rectangle(img, (current_rectangle[0], current_rectangle[1]),
                     (current_rectangle[2], current_rectangle[3]), COLOR_PALETTE[current_type], 2)
    
    # Vertical text display with type info
    text_color = COLOR_PALETTE[current_type]
    y_start = 40  # Increased from 30 for better visibility
    line_height = 35  # Increased from 25
    
    # Current type info
    cv2.putText(img, f"Type: {current_type}", (20, y_start), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)
    cv2.putText(img, f"ROIs: {len(rectangles[current_type])}", (20, y_start + line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)
    
    # All types legend (right side)
    legend_x = img.shape[1] - 200  # Adjusted for higher resolution
    cv2.putText(img, "Fuse Types:", (legend_x, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    for i, (t, c) in enumerate(COLOR_PALETTE.items()):
        y_pos = 80 + (i * 35)  # Increased spacing
        cv2.putText(img, f"{t}: ", (legend_x, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.circle(img, (legend_x + 50, y_pos - 5), 7, c, -1)  # Larger circles
    
    cv2.imshow("Video", img)

    key = cv2.waitKey(1) & 0xFF
    if 49 <= key <= 56:
        current_type = key - 48
    
    if key == ord('s'):
        with open("fuse_rois.pkl", "wb") as f:
            pickle.dump(rectangles, f)
        print("ROIs saved.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()