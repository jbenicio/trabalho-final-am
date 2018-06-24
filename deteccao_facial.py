import cv2
import dlib

imagem = cv2.imread("recursos/iron.jpg")
detector = dlib.cnn_face_detection_model_v1("recursos/mmod_human_face_detector.dat")
facesDetectadas = detector(imagem, 2)
for face in facesDetectadas:
    e, t, d, b, c = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), int(face.rect.bottom()), face.confidence)
    cv2.rectangle(imagem, (e, t), (d, b), (255, 255, 0), 2)

cv2.imshow("Detector CNN", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()