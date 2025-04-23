import cv2
import pickle

class ROISelector:
    def __init__(self):
        self.rois = []
        self.start_point = None
        self.drawing = False

        self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow("ROI Selector")
        cv2.setMouseCallback("ROI Selector", self.mouse_events)

    def mouse_events(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = self.start_point
            x2, y2 = x, y
            rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.rois.append(rect)
            print(f"Added ROI: {rect}")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            display_frame = frame.copy()

            # Draw existing ROIs
            for idx, rect in enumerate(self.rois):
                cv2.rectangle(display_frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
                cv2.putText(display_frame, f"ROI {idx+1}", (rect[0], rect[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Draw currently drawing rectangle
            if self.drawing and hasattr(self, 'temp_end'):
                cv2.rectangle(display_frame, self.start_point, self.temp_end, (0, 255, 0), 1)

            cv2.putText(display_frame, "Draw ROIs with mouse", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display_frame, "Press 's' to save, 'q' to quit", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.imshow("ROI Selector", display_frame)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key == ord('s'):
                with open("rois.pkl", "wb") as f:
                    pickle.dump(self.rois, f)
                print("ROIs saved to rois.pkl")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    selector = ROISelector()
    selector.run()
