import cv2
import numpy as np
user_img = cv2.imread("user.jpg")
if user_img is None:
    print("Error: user.jpg not found or can't be loaded")
    exit()
user_img = cv2.resize(user_img, (640, 480))
gray = cv2.cvtColor(user_img, cv2.COLOR_BGR2GRAY)
sunglass = cv2.imread("sunglass.png", cv2.IMREAD_UNCHANGED)
if sunglass is None or sunglass.shape[2] != 4:
    print("Error: sunglass.png not found or does not have alpha channel")
    exit()
scale_factor = 1.5
new_w = int(sunglass.shape[1] * scale_factor)
new_h = int(sunglass.shape[0] * scale_factor)
sunglass = cv2.resize(sunglass, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print("Faces detected:", len(faces))
for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = user_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
    print("Eyes detected:", len(eyes))
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
        eye_centers = []
        for (ex, ey, ew, eh) in eyes:
            center_x = x + ex + ew // 2
            center_y = y + ey + eh // 2
            eye_centers.append((center_x, center_y))
        eye_centers = sorted(eye_centers, key=lambda c: c[0])
        left_eye = eye_centers[0]
        right_eye = eye_centers[1]
        eye_distance = right_eye[0] - left_eye[0]
        glasses_width = int(2.2 * eye_distance)
        scaling_factor = glasses_width / sunglass.shape[1]
        glasses_height = int(sunglass.shape[0] * scaling_factor)
        resized_glasses = cv2.resize(sunglass, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
        x_offset = int((left_eye[0] + right_eye[0]) / 2 - glasses_width / 2)
        y_offset = int(min(left_eye[1], right_eye[1]) - glasses_height * 0.55)
        x1, x2 = max(x_offset, 0), min(x_offset + glasses_width, user_img.shape[1])
        y1, y2 = max(y_offset, 0), min(y_offset + glasses_height, user_img.shape[0])
        glasses_x1 = 0 if x_offset >= 0 else -x_offset
        glasses_x2 = glasses_width - (x_offset + glasses_width - x2) if x2 < x_offset + glasses_width else glasses_width
        glasses_y1 = 0 if y_offset >= 0 else -y_offset
        glasses_y2 = glasses_height - (y_offset + glasses_height - y2) if y2 < y_offset + glasses_height else glasses_height
        alpha_glasses = resized_glasses[glasses_y1:glasses_y2, glasses_x1:glasses_x2, 3] / 255.0
        alpha_background = 1.0 - alpha_glasses
        for c in range(3):
            user_img[y1:y2, x1:x2, c] = (alpha_glasses * resized_glasses[glasses_y1:glasses_y2, glasses_x1:glasses_x2, c] +
                                         alpha_background * user_img[y1:y2, x1:x2, c]).astype(np.uint8)

cv2.imshow("Try-On Result", user_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output_tryon.jpg", user_img)