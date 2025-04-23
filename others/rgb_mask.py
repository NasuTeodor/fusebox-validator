import cv2
import numpy as np
import pickle

# Încărcarea coordonatelor ROI
with open('rectangles.pkl', 'rb') as f:
    roi_coordinates = pickle.load(f)

start_x, start_y, end_x, end_y = roi_coordinates[0]

# Inițializare captură video cu DSHOW
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Conexiune la camera eșuată.")

# Configurare fereastră de control
cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Controls', 600, 350)

# Trackbar-uri pentru RGB și blur
cv2.createTrackbar('R Min', 'Controls', 0, 255, lambda x: None)
cv2.createTrackbar('R Max', 'Controls', 255, 255, lambda x: None)
cv2.createTrackbar('G Min', 'Controls', 0, 255, lambda x: None)
cv2.createTrackbar('G Max', 'Controls', 255, 255, lambda x: None)
cv2.createTrackbar('B Min', 'Controls', 0, 255, lambda x: None)
cv2.createTrackbar('B Max', 'Controls', 255, 255, lambda x: None)
cv2.createTrackbar('Blur Intensity', 'Controls', 0, 15, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Eroare la captura frame-ului")
        break

    # Extrage ROI-ul
    roi = frame[start_y:end_y, start_x:end_x]
    
    if roi.size == 0:
        print("ROI invalid!")
        break

    # Aplică blur
    blur_val = cv2.getTrackbarPos('Blur Intensity', 'Controls')
    if blur_val > 0:
        ksize = 2 * blur_val + 1
        roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)

    # Convertire la RGB
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Citire valori RGB
    r_min = cv2.getTrackbarPos('R Min', 'Controls')
    r_max = cv2.getTrackbarPos('R Max', 'Controls')
    g_min = cv2.getTrackbarPos('G Min', 'Controls')
    g_max = cv2.getTrackbarPos('G Max', 'Controls')
    b_min = cv2.getTrackbarPos('B Min', 'Controls')
    b_max = cv2.getTrackbarPos('B Max', 'Controls')

    # Creare mască RGB
    lower_rgb = np.array([r_min, g_min, b_min])
    upper_rgb = np.array([r_max, g_max, b_max])
    mask = cv2.inRange(rgb_roi, lower_rgb, upper_rgb)

    # Aplicare mască și convertire înapoi la BGR pentru afișare
    masked_rgb = cv2.bitwise_and(rgb_roi, rgb_roi, mask=mask)
    masked_bgr = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)

    # Afișare rezultate
    cv2.imshow('Original ROI', roi)
    cv2.imshow('Masked ROI', masked_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()