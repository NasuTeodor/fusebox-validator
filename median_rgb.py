import cv2
import numpy as np
import pickle

class ROIMasker:
    def __init__(self):
        # Load ROIs
        with open("rois.pkl", "rb") as f:
            self.rois = pickle.load(f)

        # Setup camera
        self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Initialize tolerances
        self.tolerances = {
            'R_low': 0, 'R_high': 0,
            'G_low': 0, 'G_high': 0,
            'B_low': 0, 'B_high': 0
        }

        # Create sliders
        cv2.namedWindow("Controls")
        for color in ['R', 'G', 'B']:
            cv2.createTrackbar(f"{color} Lower", "Controls", 0, 255, lambda v, c=color: self.update_tolerance(f"{c}_low", v))
            cv2.createTrackbar(f"{color} Upper", "Controls", 0, 255, lambda v, c=color: self.update_tolerance(f"{c}_high", v))

    def update_tolerance(self, key, value):
        self.tolerances[key] = value

    def get_median_rgb(self, frame):
        pixels = []
        for x1, y1, x2, y2 in self.rois:
            roi = frame[y1:y2, x1:x2]
            pixels.extend(roi.reshape(-1, 3))
        if pixels:
            median = np.median(np.array(pixels), axis=0).astype(np.uint8)
            return median
        return np.array([0, 0, 0], dtype=np.uint8)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            median_color = self.get_median_rgb(frame)
            # Convert to int to avoid overflow when doing arithmetic
            b, g, r = map(int, median_color)

            # Apply individual lower and upper tolerances
            lower = np.array([
                max(0, b - self.tolerances['B_low']),
                max(0, g - self.tolerances['G_low']),
                max(0, r - self.tolerances['R_low'])
            ], dtype=np.uint8)

            upper = np.array([
                min(255, b + self.tolerances['B_high']),
                min(255, g + self.tolerances['G_high']),
                min(255, r + self.tolerances['R_high'])
            ], dtype=np.uint8)

            # Create mask and masked image
            mask = cv2.inRange(frame, lower, upper)
            masked_result = cv2.bitwise_and(frame, frame, mask=mask)

            # Display results
            cv2.imshow("Original", frame)
            cv2.imshow("Mask", masked_result)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ROIMasker().run()
