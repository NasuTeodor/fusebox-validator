import cv2
import numpy as np
import pickle
import sys

class MultiROIAnalyzer:
    def __init__(self):
        self.rois = self.load_rois()
        self.mask_ranges = self.load_mask_ranges()
        self.base_valid_count = [85] * 10
        self.current_type = 1

        self.colors = {
            1: (255, 0, 0),   2: (0, 255, 0),   3: (0, 0, 255),   4: (255, 255, 0),
            5: (255, 0, 255), 6: (0, 255, 255), 7: (128, 0, 255), 8: (128, 255, 128)
        }

        # Initialize camera
        self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow('Controls')
        cv2.namedWindow('Camera Feed')
        # cv2.namedWindow('Masks')

        for i in range(1, 9):
            cv2. createTrackbar(f'Threshold{i}', 'Controls',
                                self.base_valid_count[i], 100, 
                                lambda v,t=i: self.update_base_valid_count(t, v)
                                )

    def update_base_valid_count(self, type, value):
        self.base_valid_count[type] = value

    def load_rois(self):
        try:
            with open("fuse_rois.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Error: 'fuse_rois.pkl' not found.")
            sys.exit(1)

    def load_mask_ranges(self):
        try:
            with open("hsv_tolerances.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Error: 'hsv_tolerances.pkl' not found.")
            sys.exit(1)

    def save_ranges(self):
        fusebox_data = {}
        mask_ranges = self.mask_ranges
        rois = self.rois
        thresholds = self.base_valid_count
        fusebox_data = {
            "rois": rois,
            "mask_ranges": mask_ranges,
            "thresholds": thresholds
        }
        with open("fusebox.pkl", "wb") as f:
            pickle.dump(fusebox_data, f)
        print("Fusebox data saved!")

    def compute_mask(self, type, hsv):
        mask_range = self.mask_ranges[type]
        lower, upper = mask_range['lower'], mask_range['upper']
        mask = cv2.inRange(hsv, lower, upper)
        return mask
    
    def validate_type(self, type, frame):
        valid_list = []
        frame_copy = frame.copy()
        mask = self.compute_mask(type, frame_copy)
        rois = self.rois[type]

        for roi in rois:
            x1, y1, x2, y2 = roi
            fuse = mask[y1:y2, x1:x2]
            pixels = fuse.size
            white = cv2.countNonZero(fuse)
            # print(f'{pixels}-{white}', end='\r')
            perc = (white / pixels) * 100
            valid_list.append(perc)
        return valid_list

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # mask = self.compute_mask(self.current_type, hsv)
            # computed = cv2.bitwise_and(frame, frame, mask=mask)

            frame_copy = frame.copy()
            for type in range(1, 9):
                valid_list = self.validate_type(type, hsv)
                rois = self.rois[type]
                i = 0
                for roi in rois:
                    x1, y1, x2, y2 = roi
                    status_color = (0, 255, 0)
                    if int(valid_list[i]) <= self.base_valid_count[type]:
                        status_color = (0, 0, 255)
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), status_color, 2)
                    cv2.putText(frame_copy, f"{type}:{int(valid_list[i])}%", (x1, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    i+=1

            # rois = self.rois[self.current_type]
            # for roi in rois:
            #     x1, y1, x2, y2 = roi
            #     cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # cv2.imshow('ROI', valid)
            # cv2.imshow('Masks', computed)
            cv2.imshow('Camera Feed', frame_copy)

            key = cv2.waitKey(100)
            if key == ord('q'):
                break
            elif 49 <= key <= 56:  # Keys 1 to 8
                self.current_type = key - 48
            elif key == ord('s'):
                self.save_ranges()
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MultiROIAnalyzer()
    app.run()