import cv2
import numpy as np
import pickle

class MultiROIAnalyzer:
    def __init__(self):
        # Load data
        self.rois = self.load_rois()
        self.mask_ranges = self.load_mask_ranges()
        self.thresholds = {i: 30 for i in range(1, 9)}
        
        # Color palette for different types
        self.colors = {
            1: (255, 0, 0),   2: (0, 255, 0),   3: (0, 0, 255),   4: (255, 255, 0),
            5: (255, 0, 255), 6: (0, 255, 255), 7: (128, 0, 255), 8: (128, 255, 128)
        }

        # Initialize camera
        self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create GUI
        cv2.namedWindow('Camera Feed')
        cv2.namedWindow('Masks')
        cv2.namedWindow('Controls')
        self.create_controls()

    def create_controls(self):
        # Create threshold sliders for all types
        for i in range(1, 9):
            cv2.createTrackbar(f'T{i} Threshold', 'Controls', 
                              self.thresholds[i], 100, 
                              lambda v,t=i: self.update_threshold(t, v))

    def update_threshold(self, type_idx, value):
        self.thresholds[type_idx] = value

    def load_rois(self):
        try:
            with open("fuse_rois.pkl", "rb") as f:
                saved_rois = pickle.load(f)
            return self.process_rois(saved_rois)
        except FileNotFoundError:
            return {i: [] for i in range(1, 9)}

    def process_rois(self, saved_dict):
        return {t: [{'coords': rect, 'id': idx+1} 
                for idx, rect in enumerate(rects)] 
                for t, rects in saved_dict.items()}

    def load_mask_ranges(self):
        try:
            with open("mask_ranges.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {i: {"low": [0,0,0], "high": [255,255,255]} for i in range(1,9)}
        
    def calculate_accuracy(self, results):
        total_valid = 0
        total_rois = 0
        threshold_sum = 0
        
        for t in range(1, 9):
            type_data = results.get(t, {})
            rois = type_data.get('rois', [])
            general_median = type_data.get('general_median')
            threshold = self.thresholds[t]
            
            valid_count = 0
            for roi in rois:
                if roi['median'] is not None and general_median is not None:
                    distance = self.color_distance(roi['median'], general_median)
                    if distance <= threshold:
                        valid_count += 1
            
            total_valid += valid_count
            total_rois += len(rois)
            threshold_sum += threshold
        
        if total_rois == 0:
            return 0.0, 0.0
        
        validation_ratio = total_valid / total_rois
        avg_threshold = threshold_sum / 8
        accuracy = validation_ratio * (1 - avg_threshold/100) * 100
        
        return accuracy, avg_threshold

    def calculate_medians(self, frame):
        results = {}
        for t in range(1, 9):
            mask = cv2.inRange(frame, 
                             np.array(self.mask_ranges[t]['low']),
                             np.array(self.mask_ranges[t]['high']))
            
            type_rois = []
            all_pixels = []
            
            for roi in self.rois.get(t, []):
                x1, y1, x2, y2 = roi['coords']
                roi_region = frame[y1:y2, x1:x2]
                roi_mask = mask[y1:y2, x1:x2]
                
                # Extract non-zero pixels
                pixels = roi_region[roi_mask > 0]
                median = np.median(pixels, axis=0) if len(pixels) > 0 else None
                
                type_rois.append({
                    'coords': (x1, y1, x2, y2),
                    'median': median,
                    'id': roi['id']
                })
                
                if median is not None:
                    all_pixels.extend(pixels)
            
            general_median = np.median(all_pixels, axis=0) if len(all_pixels) > 0 else None
            results[t] = {'rois': type_rois, 'general_median': general_median}
        
        return results

    def color_distance(self, c1, c2):
        return np.linalg.norm(c1 - c2) if (c1 is not None and c2 is not None) else float('inf')

    def update_display(self, frame, results):
        display_frame = frame.copy()
        mask_display = np.zeros_like(frame)
        
        for t in range(1, 9):
            type_data = results.get(t, {})
            general_median = type_data.get('general_median')
            color = self.colors[t]
            
            # Create mask overlay
            mask = cv2.inRange(frame, 
                              np.array(self.mask_ranges[t]['low']),
                              np.array(self.mask_ranges[t]['high']))
            mask_display[mask > 0] = color
            
            # Draw ROIs
            for roi in type_data.get('rois', []):
                x1, y1, x2, y2 = roi['coords']
                status_color = color
                
                if general_median is not None and roi['median'] is not None:
                    distance = self.color_distance(roi['median'], general_median)
                    if distance > self.thresholds[t]:
                        status_color = tuple(c//4 for c in color)  # Darker for inactive
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), status_color, 2)
                cv2.putText(display_frame, f"{t}-{roi['id']}", (x1+5, y1+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Combine mask display with alpha
        mask_display = cv2.addWeighted(frame, 0.7, mask_display, 0.3, 0)

        # Calculate and display accuracy
        accuracy, avg_threshold = self.calculate_accuracy(results)
        
        # Draw accuracy information
        info_y = 30
        cv2.putText(display_frame, f"System Accuracy: {accuracy:.1f}%", 
                   (10, info_y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(display_frame, f"Avg Threshold: {avg_threshold:.1f}", 
                   (10, info_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        # Show displays
        cv2.imshow('Camera Feed', display_frame)
        cv2.imshow('Masks', mask_display)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            results = self.calculate_medians(frame)
            self.update_display(frame, results)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                with open("thresholds.pkl", "wb") as f:
                    pickle.dump(self.thresholds, f)
                print("Thresholds saved to thresholds.pkl")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = MultiROIAnalyzer()
    analyzer.run()