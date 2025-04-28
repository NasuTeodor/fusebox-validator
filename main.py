import cv2, pickle, sys, time, os
from thingworx import get_thing_properties

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Result')
cv2.namedWindow('Last Validation')
# cv2.namedWindow('Tolerances')
# cv2.resizeWindow('Tolerances', 640, 360)

validation_wait_time = 5
average_accepted_validation = 50 #50% of all verified time

should_stop = False
while not should_stop:
    result = get_thing_properties()['boxName']
    print(f"Validating {result}")

    dir_list = os.listdir()
    selected_box = None
    for file in dir_list:
        if str.lower(file[:-4]) == str.lower(result):
            selected_box = file
    
    if selected_box == None:
        continue

    try:
        with open(selected_box, "rb") as f:
            fusebox_data = pickle.load(f)
            f.close()
    except FileNotFoundError:
        print("Fusebox data file not found!")

    # def update_thresholds(type, value):
    #     fusebox_data['thresholds'][type] = value
    # for i in range(1, 9):
    #     cv2. createTrackbar(f'Threshold{i}', 'Tolerances',
    #                         fusebox_data['thresholds'][i], 100, 
    #                         lambda v,t=i: update_thresholds(t, v)
    #                         )

    validation_start_time = time.time()
    frame_count = 0
    validation_stat = {type: {} for type in range(1, 9)}
    
    while time.time() - validation_start_time <= validation_wait_time:
        ret, frame = cap.read()
        if not ret:
            print("Camera said bye")
            break
        frame_count += 1

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        frame_copy = frame.copy()
        for type in range(1, 9):
            rois = fusebox_data['rois'][type]
            mask_ranges = fusebox_data['mask_ranges'][type]
            threshold = fusebox_data['thresholds'][type]

            lower = mask_ranges['lower']
            upper = mask_ranges['upper']
            mask = cv2.inRange(hsv_frame, lowerb=lower, upperb=upper)


            # Create text
            range_text = f'Type {type}: lowerB={lower} - upperB={upper} | {threshold}'

            # Choose position for each type, stacked vertically
            text_x = 10
            text_y = 30 * type  # 30 pixels spacing between lines

            # Draw text
            cv2.putText(frame_copy, range_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # Draw circle with lower bound color
            circle_x_lower = text_x + 620  # shift it right from text
            circle_y = text_y - 5  # align vertically with text
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
            cv2.circle(frame_copy, (circle_x_lower, circle_y), 6, tuple(int(c) for c in lower), -1)

            # Draw circle with upper bound color
            circle_x_upper = circle_x_lower + 20
            cv2.circle(frame_copy, (circle_x_upper, circle_y), 6, tuple(int(c) for c in upper), -1)
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_HSV2BGR)

            for roi in rois:
                x1,y1,x2,y2 = roi
                fuse = mask[y1:y2, x1:x2]
                pixels = fuse.size
                white = cv2.countNonZero(fuse)
                perc = (white / pixels) * 100
                roi_status = (0, 0, 255)
                if perc >= threshold:
                        roi_status = (0, 255, 0)

                        if roi in validation_stat[type]:
                            validation_stat[type][roi] += 1
                        else:
                            validation_stat[type][roi] = 1
                else:
                    if roi not in validation_stat[type]:
                        validation_stat[type][roi] = 0

                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), roi_status, 2)
                cv2.putText(frame_copy, f'{type}:{int(perc)}%', (x1, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        key = cv2.waitKey(1)
        if key == ord('q'):
            should_stop = True
            break

        cv2.imshow('Result', frame_copy)
    
    frame_copy = frame.copy()
    fps = round(frame_count / (time.time() - validation_start_time), 2)
    for type in validation_stat:
        rois = fusebox_data['rois'][type]
        for roi in rois:
            x1, y1, x2, y2 = roi
            tot_validations = validation_stat[type][roi]
            avg_validations = int((tot_validations / frame_count) * 100)
            # status_color = (0, 0, 255)
            if avg_validations < average_accepted_validation:
                # status_color = (0, 255, 0)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_copy, f"Avg:{avg_validations}", (x1, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow('Last Validation', frame_copy)

cap.release()
cv2.destroyAllWindows()