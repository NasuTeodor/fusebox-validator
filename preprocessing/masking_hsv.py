import cv2
import numpy as np
import pickle

class ROIMaskCreator:
    def __init__(self):
        # Load saved ROIs
        self.rois = self.load_rois()
        self.current_type = 1
        # self.mask_ranges = self.load_tolerances()
        self.mask_ranges = {
            i: {
                'H_low': 100, 'H_high': 100,
                'S_low': 100, 'S_high': 100,
                'V_low': 100, 'V_high': 100
            } for i in range(1, 9)
        }
        self.final_ranges = {}

        # Initialize camera
        self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
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
            raise FileNotFoundError()
            # return {i: [] for i in range(1, 9)}

    # def load_tolerances(self):
    #     return {
    #         i: {
    #             'H_low': 0, 'H_high': 0,
    #             'S_low': 0, 'S_high': 0,
    #             'V_low': 0, 'V_high': 0
    #         } for i in range(1, 9)
    #     }

    def create_trackbars(self):
        for key in self.mask_ranges[self.current_type]:
            cv2.createTrackbar(key, 'Controls', 0, 100, lambda x: None)

    def update_trackbar_positions(self):
        for key in self.mask_ranges[self.current_type]:
            cv2.setTrackbarPos(key, 'Controls', self.mask_ranges[self.current_type][key])

    def update_tolerances(self):
        for key in self.mask_ranges[self.current_type]:
            self.mask_ranges[self.current_type][key] = cv2.getTrackbarPos(key, 'Controls')

    def compute_median_hsv(self, hsv_frame):
        roi_pixels = []
        for (x1, y1, x2, y2) in self.rois.get(self.current_type, []):
            roi = hsv_frame[y1:y2, x1:x2]
            roi_pixels.append(roi.reshape(-1, 3))
        if not roi_pixels:
            return np.array([0, 0, 0], dtype=int)
        all_pixels = np.vstack(roi_pixels)
        median = np.median(all_pixels, axis=0).astype(int)
        return median

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            self.update_tolerances()

            h, s, v = self.compute_median_hsv(hsv)
            tol = self.mask_ranges[self.current_type]

            lower = np.array([
                max(0, h - tol['H_low']),
                max(0, s - tol['S_low']),
                max(0, v - tol['V_low'])
            ], dtype=np.uint8)

            upper = np.array([
                min(179, h + tol['H_high']),
                min(255, s + tol['S_high']),
                min(255, v + tol['V_high'])
            ], dtype=np.uint8)

            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            self.final_ranges[self.current_type] = {
                'lower': lower,
                'upper': upper
            }

            # Draw current ROIs
            frame_disp = frame.copy()
            for roi in self.rois.get(self.current_type, []):
                x1, y1, x2, y2 = roi
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display
            cv2.imshow('Camera Feed', frame_disp)
            cv2.imshow('Mask', result)

            key = cv2.waitKey(1)
            if key == ord('s'):
                self.save_tolerances()
            elif 49 <= key <= 56:  # 1-8 keys
                self.current_type = key - 48
                self.update_trackbar_positions()
            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def save_tolerances(self):
        with open("hsv_tolerances.pkl", "wb") as f:
            # pickle.dump(self.mask_ranges, f)
            pickle.dump(self.final_ranges, f)
        print("HSV mask ranges saved!")

if __name__ == "__main__":
    app = ROIMaskCreator()
    app.run()
