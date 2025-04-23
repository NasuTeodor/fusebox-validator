import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

class StableHistogramAnalyzer:
    def __init__(self):
        # Load data and initialize camera
        self.rois = self.load_rois()
        self.mask_ranges = self.load_mask_ranges()
        self.current_type = 1
        self.prev_type = None
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Matplotlib setup
        plt.ion()
        self.fig = None
        self.axes = []
        
        # Windows
        cv2.namedWindow("Camera Feed")
        cv2.namedWindow("Masked View")

    def load_rois(self):
        try:
            with open("fuse_rois.pkl", "rb") as f:
                saved_rois = pickle.load(f)
            return self.process_rois(saved_rois)
        except FileNotFoundError:
            return []

    def process_rois(self, saved_dict):
        return [{'type': t, 'coords': rect, 'id': idx+1} 
                for t, rects in saved_dict.items() for idx, rect in enumerate(rects)]

    def load_mask_ranges(self):
        try:
            with open("mask_ranges.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {i: {"low": [0,0,0], "high": [255,255,255]} for i in range(1,9)}

    def create_mask(self, frame):
        ranges = self.mask_ranges[self.current_type]
        return cv2.inRange(frame, np.array(ranges['low']), np.array(ranges['high']))

    def update_histogram_window(self, frame, mask):
        current_rois = [r for r in self.rois if r['type'] == self.current_type]
        
        # Only recreate figure when type changes or ROI count changes
        if self.current_type != self.prev_type or \
           (self.fig and len(current_rois) != len(self.axes)):
            if self.fig:
                plt.close(self.fig)
            self.fig, self.axes = plt.subplots(len(current_rois), 1, figsize=(8, 4*len(current_rois)))
            if not isinstance(self.axes, np.ndarray):
                self.axes = [self.axes]
            self.fig.canvas.manager.set_window_title(f"Type {self.current_type} Histograms")
            plt.tight_layout()
            self.prev_type = self.current_type

        for ax, roi in zip(self.axes, current_rois):
            x1, y1, x2, y2 = roi['coords']
            roi_region = frame[y1:y2, x1:x2]
            roi_mask = mask[y1:y2, x1:x2]
            masked_roi = cv2.bitwise_and(roi_region, roi_region, mask=roi_mask)
            
            ax.clear()
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([masked_roi], [i], None, [256], [0, 256])
                ax.plot(hist, color=color, alpha=0.7)
            
            ax.set_title(f"ROI {roi['id']} - {np.count_nonzero(roi_mask)} pixels")
            ax.set_xlim([0, 256])
            ax.grid(True)

        if self.fig:
            plt.pause(0.001)
            self.fig.canvas.draw_idle()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Create mask and masked view
            mask = self.create_mask(frame)
            masked_view = cv2.bitwise_and(frame, frame, mask=mask)

            # Draw ROIs on camera feed
            frame_disp = frame.copy()
            current_rois = [r for r in self.rois if r['type'] == self.current_type]
            for roi in current_rois:
                x1, y1, x2, y2 = roi['coords']
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame_disp, str(roi['id']), (x1+5, y1+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Update displays
            cv2.imshow("Camera Feed", frame_disp)
            cv2.imshow("Masked View", masked_view)
            self.update_histogram_window(frame, mask)

            key = cv2.waitKey(1)
            if 49 <= key <= 56:  # 1-8 keys
                self.current_type = key - 48
            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        plt.close('all')

if __name__ == "__main__":
    analyzer = StableHistogramAnalyzer()
    analyzer.run()