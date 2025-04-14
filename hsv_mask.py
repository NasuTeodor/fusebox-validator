import cv2
import numpy as np
import pickle

# Încărcarea coordonatelor ROI
with open('rectangles.pkl', 'rb') as f:
    roi_coordinates = pickle.load(f)

start_x, start_y, end_x, end_y = roi_coordinates[0]

# Inițializare captură video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Nu pot conecta la camera 2. Verifică conexiunea.")

# Fereastra de controale
cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Controls', 600, 350)

# Trackbar-uri pentru HSV și blur
cv2.createTrackbar('H Min', 'Controls', 0, 255, lambda x: None)
cv2.createTrackbar('H Max', 'Controls', 179, 255, lambda x: None)
cv2.createTrackbar('S Min', 'Controls', 0, 255, lambda x: None)
cv2.createTrackbar('S Max', 'Controls', 255, 255, lambda x: None)
cv2.createTrackbar('V Min', 'Controls', 0, 255, lambda x: None)
cv2.createTrackbar('V Max', 'Controls', 255, 255, lambda x: None)
cv2.createTrackbar('Blur Intensity', 'Controls', 0, 100, lambda x: None)  # 0-15 => kernel size 1-31

while True:
    ret, frame = cap.read()
    if not ret:
        print("Eroare la citirea frame-ului")
        break
    
    # Extrage ROI
    roi = frame[start_y:end_y, start_x:end_x]
    
    if roi.size == 0:
        print("ROI invalid!")
        break

    # Aplică blur dacă e necesar
    blur_intensity = cv2.getTrackbarPos('Blur Intensity', 'Controls')
    if blur_intensity > 0:
        kernel_size = 2 * blur_intensity + 1  # Convertire 0-15 -> 1-31 (numere impare)
        roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)

    # Procesare HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Citire valori HSV
    h_min = cv2.getTrackbarPos('H Min', 'Controls')
    h_max = cv2.getTrackbarPos('H Max', 'Controls')
    s_min = cv2.getTrackbarPos('S Min', 'Controls')
    s_max = cv2.getTrackbarPos('S Max', 'Controls')
    v_min = cv2.getTrackbarPos('V Min', 'Controls')
    v_max = cv2.getTrackbarPos('V Max', 'Controls')
    
    # Creare și aplicare mască
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_roi, lower, upper)
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)

    # Afișare
    cv2.imshow('Original ROI', roi)
    cv2.imshow('Masked ROI', masked_roi)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()