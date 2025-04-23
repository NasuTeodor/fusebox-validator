import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Configuration
ROI_FILE = "fuse_rois.pkl"
COLOR_PALETTE = {
    1: (255, 0, 0),   2: (0, 255, 0),   3: (0, 0, 255),   4: (255, 255, 0),
    5: (255, 0, 255), 6: (0, 255, 255), 7: (128, 0, 255), 8: (128, 255, 128)
}
HIST_BINS = 64

class FixedHistogramAnalyzer:
    def __init__(self):
        # Camera setup
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # ROI management
        self.rois = self.load_rois()
        self.current_type = 1
        self.fig = None
        self.axes = []
        self.lines = []
        
        # Matplotlib setup
        plt.ion()

    def load_rois(self):
        try:
            with open(ROI_FILE, "rb") as f:
                saved_data = pickle.load(f)
            return self.process_roi_data(saved_data)
        except FileNotFoundError:
            return []

    def process_roi_data(self, saved_dict):
        rois = []
        for t, rects in saved_dict.items():
            for idx, rect in enumerate(rects, 1):
                rois.append({
                    'type': t,
                    'coords': rect,
                    'id': idx,
                    'color': COLOR_PALETTE[t]
                })
        return rois

    def close_histogram_window(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = []
            self.lines = []

    def create_histogram_window(self):
        self.close_histogram_window()
        
        current_rois = [r for r in self.rois if r['type'] == self.current_type]
        n_rois = len(current_rois)
        
        if n_rois == 0:
            return

        cols = min(3, n_rois)
        rows = (n_rois + cols - 1) // cols
        
        self.fig, self.axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
        if not isinstance(self.axes, np.ndarray):
            self.axes = np.array([self.axes])
        self.axes = self.axes.flatten()
        
        self.fig.canvas.manager.set_window_title(f"Type {self.current_type} Histograms")
        plt.tight_layout(pad=3.0)
        
        self.lines = []
        for ax in self.axes:
            ax.clear()
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 1)
            ax.grid(True)
            ax.set_xlabel("Intensity")
            lines = [
                ax.plot([], [], 'r', alpha=0.5)[0],  # Red
                ax.plot([], [], 'g', alpha=0.5)[0],  # Green
                ax.plot([], [], 'b', alpha=0.5)[0]   # Blue
            ]
            self.lines.append(lines)
        
        # Remove extra axes if any
        for ax in self.axes[n_rois:]:
            ax.remove()
        
        plt.draw()
        plt.pause(0.01)

    def update_histograms(self, frame):
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            return

        current_rois = [r for r in self.rois if r['type'] == self.current_type]
        bin_edges = np.linspace(0, 255, HIST_BINS)
        
        for idx, roi in enumerate(current_rois):
            if idx >= len(self.axes):
                break

            x1, y1, x2, y2 = roi['coords']
            roi_img = frame[y1:y2, x1:x2]
            
            if roi_img.size == 0:
                continue

            ax = self.axes[idx]
            ax.set_title(f"ROI {roi['id']}")
            
            for ch in range(3):
                hist = cv2.calcHist([roi_img], [ch], None, [HIST_BINS], [0, 256])
                hist = cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX)
                self.lines[idx][ch].set_data(bin_edges, hist)
            
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def run(self):
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Update ROI display
                display_frame = frame.copy()
                current_rois = [r for r in self.rois if r['type'] == self.current_type]
                for roi in current_rois:
                    x1, y1, x2, y2 = roi['coords']
                    color = tuple(c//2 for c in roi['color'])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"{roi['id']}", (x1+5, y1+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Update histograms
                self.update_histograms(frame)
                cv2.imshow("Camera Feed", display_frame)

                key = cv2.waitKey(1)
                if 49 <= key <= 56:  # Number keys 1-8
                    self.current_type = key - 48
                    self.create_histogram_window()
                elif key == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.close_histogram_window()
            plt.close('all')

if __name__ == "__main__":
    analyzer = FixedHistogramAnalyzer()
    analyzer.run()