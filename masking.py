import cv2
import numpy as np
import pickle

class ROIMaskCreator:
    def __init__(self):
        # Load saved ROIs
        self.rois = self.load_rois()
        self.current_type = 1
        self.mask_ranges = {i: {"low": [0, 0, 0], "high": [255, 255, 255]} for i in range(1, 9)}
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create GUI windows
        cv2.namedWindow('Camera Feed')
        cv2.namedWindow('Mask')
        cv2.namedWindow('Controls')
        
        # Create trackbars
        self.create_trackbars()
        self.update_trackbar_positions()
        
    def load_rois(self):
        try:
            with open("fuse_rois.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {i: [] for i in range(1, 9)}
    
    def create_trackbars(self):
        cv2.createTrackbar('Low B', 'Controls', 0, 255, lambda x: self.update_range(0, 'low', x))
        cv2.createTrackbar('High B', 'Controls', 255, 255, lambda x: self.update_range(0, 'high', x))
        cv2.createTrackbar('Low G', 'Controls', 0, 255, lambda x: self.update_range(1, 'low', x))
        cv2.createTrackbar('High G', 'Controls', 255, 255, lambda x: self.update_range(1, 'high', x))
        cv2.createTrackbar('Low R', 'Controls', 0, 255, lambda x: self.update_range(2, 'low', x))
        cv2.createTrackbar('High R', 'Controls', 255, 255, lambda x: self.update_range(2, 'high', x))
    
    def update_range(self, channel, range_type, value):
        self.mask_ranges[self.current_type][range_type][channel] = value
    
    def update_trackbar_positions(self):
        ranges = self.mask_ranges[self.current_type]
        cv2.setTrackbarPos('Low B', 'Controls', ranges['low'][0])
        cv2.setTrackbarPos('High B', 'Controls', ranges['high'][0])
        cv2.setTrackbarPos('Low G', 'Controls', ranges['low'][1])
        cv2.setTrackbarPos('High G', 'Controls', ranges['high'][1])
        cv2.setTrackbarPos('Low R', 'Controls', ranges['low'][2])
        cv2.setTrackbarPos('High R', 'Controls', ranges['high'][2])
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Draw current ROIs
            frame_disp = frame.copy()
            for roi in self.rois.get(self.current_type, []):
                x1, y1, x2, y2 = roi
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create mask
            low = np.array(self.mask_ranges[self.current_type]['low'], dtype=np.uint8)
            high = np.array(self.mask_ranges[self.current_type]['high'], dtype=np.uint8)
            mask = cv2.inRange(frame, low, high)
            
            # Display
            cv2.imshow('Camera Feed', frame_disp)
            cv2.imshow('Mask', mask)
            
            key = cv2.waitKey(1)
            if key == ord('s'):
                self.save_mask_ranges()
            elif 49 <= key <= 56:  # 1-8 keys
                self.current_type = key - 48
                self.update_trackbar_positions()
            elif key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def save_mask_ranges(self):
        with open("mask_ranges.pkl", "wb") as f:
            pickle.dump(self.mask_ranges, f)
        print(f"Mask ranges saved for all types!")

if __name__ == "__main__":
    app = ROIMaskCreator()
    app.run()