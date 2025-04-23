import cv2
import pickle
import numpy as np

class ROIMaskerHSV:
    def __init__(self, roi_file):
        self.roi_file = roi_file
        self.rois = self.load_rois()
        self.tolerances = {
            'H_low': 0, 'H_high': 0,
            'S_low': 0, 'S_high': 0,
            'V_low': 0, 'V_high': 0
        }

    def load_rois(self):
        with open(self.roi_file, 'rb') as f:
            return pickle.load(f)

    def nothing(self, x):
        pass

    def create_sliders(self):
        cv2.namedWindow('Mask')
        cv2.createTrackbar('H_low', 'Mask', 0, 100, self.nothing)
        cv2.createTrackbar('H_high', 'Mask', 0, 100, self.nothing)
        cv2.createTrackbar('S_low', 'Mask', 0, 100, self.nothing)
        cv2.createTrackbar('S_high', 'Mask', 0, 100, self.nothing)
        cv2.createTrackbar('V_low', 'Mask', 0, 100, self.nothing)
        cv2.createTrackbar('V_high', 'Mask', 0, 100, self.nothing)

    def update_tolerances(self):
        for key in self.tolerances:
            self.tolerances[key] = cv2.getTrackbarPos(key, 'Mask')

    def compute_median_hsv(self, hsv_frame):
        roi_pixels = []
        for (x1, y1, x2, y2) in self.rois:
            roi = hsv_frame[y1:y2, x1:x2]
            roi_pixels.append(roi.reshape(-1, 3))
        all_pixels = np.vstack(roi_pixels)
        median = np.median(all_pixels, axis=0).astype(int)
        return median

    def run(self):
        cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.create_sliders()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            self.update_tolerances()

            h, s, v = self.compute_median_hsv(hsv)

            lower = np.array([
                max(0, h - self.tolerances['H_low']),
                max(0, s - self.tolerances['S_low']),
                max(0, v - self.tolerances['V_low'])
            ], dtype=np.uint8)

            upper = np.array([
                min(179, h + self.tolerances['H_high']),
                min(255, s + self.tolerances['S_high']),
                min(255, v + self.tolerances['V_high'])
            ], dtype=np.uint8)

            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            # Draw ROIs for reference
            for (x1, y1, x2, y2) in self.rois:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', result)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    ROIMaskerHSV('rois.pkl').run()
