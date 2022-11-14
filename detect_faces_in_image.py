# Description: This program detects faces and eyes

# import the open CV (computer Vision) Library
import cv2

# The Haar Classifiers
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

# Read the image
img = cv2.imread("data/faces/obama_4.jpg")
faces = face_cascade.detectMultiScale(img, 1.1, 4)

# print the number of faces found
print('Faces found: ', len(faces))
print('The image height, width and channel: ', img.shape)
print('The coordinates of each faces detected: ', faces)

# loop through all the coordinates faces and draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_face = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_face, 1.1, 2)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_face, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

# write on image
font = cv2.FONT_HERSHEY_SIMPLEX
text = cv2.putText(img, 'Face Detected', (0, img.shape[0]), font, 1, (0, 255, 255), 2)

# show image
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
