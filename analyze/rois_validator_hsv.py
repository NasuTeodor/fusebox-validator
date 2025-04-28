import cv2
import numpy as np
import pickle
import sys

class MultiROIAnalyzer:
    def __init__(self):
        self.rois = self.load_rois()
        self.tolerances = self.load_tolerances()
        self.current_type = 1

        # Color palette for different types
        self.colors = {
            1: (255, 0, 0),   2: (0, 255, 0),   3: (0, 0, 255),   4: (255, 255, 0),
            5: (255, 0, 255), 6: (0, 255, 255), 7: (128, 0, 255), 8: (128, 255, 128)
        }

        # Initialize camera
        self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Create GUI windows
        cv2.namedWindow('Combined Mask')
        cv2.namedWindow('Controls')
        cv2.namedWindow('ROI Highlight')
        # self.create_trackbars()
        # self.update_trackbar_positions()

    def load_rois(self):
        try:
            with open("fuse_rois.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Error: 'fuse_rois.pkl' not found.")
            sys.exit(1)

    def load_tolerances(self):
        try:
            with open("hsv_tolerances.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Error: 'hsv_tolerances.pkl' not found.")
            sys.exit(1)

    def create_trackbars(self):
        for key in ['H_low', 'H_high', 'S_low', 'S_high', 'V_low', 'V_high']:
            cv2.createTrackbar(key, 'Controls', 0, 100, lambda x: None)

    def update_trackbar_positions(self):
        tol = self.tolerances[self.current_type]
        for key in tol:
            cv2.setTrackbarPos(key, 'Controls', tol[key])

    def update_tolerances(self):
        tol = self.tolerances[self.current_type]
        for key in tol:
            tol[key] = cv2.getTrackbarPos(key, 'Controls')

    def compute_mask_for_type(self, hsv_img, roi_type):
        h_roi_vals, s_roi_vals, v_roi_vals = [], [], []
        roi_list = self.rois.get(roi_type, [])

        for (x1, y1, x2, y2) in roi_list:
            roi = hsv_img[y1:y2, x1:x2]
            roi = roi.reshape(-1, 3)
            h_roi_vals.extend(roi[:, 0])
            s_roi_vals.extend(roi[:, 1])
            v_roi_vals.extend(roi[:, 2])

        if not h_roi_vals:
            return np.zeros(hsv_img.shape[:2], dtype=np.uint8), 0.0

        h_med = int(np.median(h_roi_vals))
        s_med = int(np.median(s_roi_vals))
        v_med = int(np.median(v_roi_vals))

        tol = self.tolerances[roi_type]

        lower = np.array([
            max(0, h_med - tol['H_low']),
            max(0, s_med - tol['S_low']),
            max(0, v_med - tol['V_low'])
        ], dtype=np.uint8)

        upper = np.array([
            min(179, h_med + tol['H_high']),
            min(255, s_med + tol['S_high']),
            min(255, v_med + tol['V_high'])
        ], dtype=np.uint8)

        mask = cv2.inRange(hsv_img, lower, upper)
        roi_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)

        for (x1, y1, x2, y2) in roi_list:
            roi_mask[y1:y2, x1:x2] = 255

        masked_area = cv2.bitwise_and(mask, mask, mask=roi_mask)
        roi_pixels = np.count_nonzero(roi_mask)
        matched_pixels = np.count_nonzero(masked_area)

        accuracy = (matched_pixels / roi_pixels) * 100 if roi_pixels else 0

        return mask, accuracy

    def draw_highlighted_rois(self, frame, hsv):
        tol = self.tolerances[self.current_type]
        h_roi_vals, s_roi_vals, v_roi_vals = [], [], []
        roi_list = self.rois.get(self.current_type, [])

        for (x1, y1, x2, y2) in roi_list:
            roi = hsv[y1:y2, x1:x2]
            roi = roi.reshape(-1, 3)
            h_roi_vals.extend(roi[:, 0])
            s_roi_vals.extend(roi[:, 1])
            v_roi_vals.extend(roi[:, 2])

        if not h_roi_vals:
            return frame

        h_med = int(np.median(h_roi_vals))
        s_med = int(np.median(s_roi_vals))
        v_med = int(np.median(v_roi_vals))

        lower = np.array([
            max(0, h_med - tol['H_low']),
            max(0, s_med - tol['S_low']),
            max(0, v_med - tol['V_low'])
        ], dtype=np.uint8)

        upper = np.array([
            min(179, h_med + tol['H_high']),
            min(255, s_med + tol['S_high']),
            min(255, v_med + tol['V_high'])
        ], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        highlighted = frame.copy()

        for (x1, y1, x2, y2) in roi_list:
            roi_mask = mask[y1:y2, x1:x2]
            if np.count_nonzero(roi_mask) > 0:
                cv2.rectangle(highlighted, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return highlighted

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # self.update_tolerances()
            accuracies = {}
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for i in range(1, 9):
                mask, acc = self.compute_mask_for_type(hsv, i)
                accuracies[i] = acc
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            highlighted_frame = self.draw_highlighted_rois(frame, hsv)

            cv2.imshow("Combined Mask", combined_mask)
            cv2.imshow("ROI Highlight", highlighted_frame)

            median_acc = 0
            for i, acc in accuracies.items():
                median_acc += acc
            median_acc /= len(accuracies)+1
            median_acc = round(median_acc, 2)
            print("--- Accuracy by Type ---")
            for i, acc in accuracies.items():
                print(f"Type {i}: {acc:.2f}%")
            print(f'General Accuracy: {median_acc}', end='\033[2J')

            key = cv2.waitKey(100)
            if key == ord('q'):
                break
            elif 49 <= key <= 56:  # Keys 1 to 8
                self.current_type = key - 48
                # self.update_trackbar_positions()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MultiROIAnalyzer()
    app.run()
