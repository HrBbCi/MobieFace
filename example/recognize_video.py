import numpy as np


def recognize_face(embedding, embeddings, labels, threshold=0.5):
    distances = np.linalg.norm(embeddings - embedding, axis=1)
    argmin = np.argmin(distances)
    minDistance = distances[argmin]

    if minDistance > threshold:
        label = "Unknown"
    else:
        label = labels[argmin]

    return (label, minDistance)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

if __name__ == "__main__":
    import cv2
    #    import argparse
    import extractors as extc
    import detectors as dt
    import pickle as cPickle
    import dlib
    import pafy as pf
    import math

    embeddings = np.load("face_embeddings.npy")
    labels = cPickle.load(open("labels.cpickle", 'rb'))
    shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
    # url = "https://www.youtube.com/watch?v=MOwaUlXZxkI" #GOT white walker Intro

    #    url = "https://www.youtube.com/watch?v=aB_UvmPqXTg"
    #    url = "https://www.youtube.com/watch?v=81Uz1mF09O4" #GOT Jamie and Cersei Lannister
    #     url = "https://www.youtube.com/watch?v=pFk2t3E2aWE"
    url = 'videos/a.mp4'  # GOT white walker Intro
    #    url = "https://www.youtube.com/watch?v=9QbltzIUV6w" # Avengers
    # url = "https://www.youtube.com/watch?v=jOJbXvjZ-cQ" # GOT white walker Intro
    #  videoPafy = pf.new(url)
    #  best = videoPafy.getbest(preftype="webm")
    #  print (videoPafy.title)
    input_video = cv2.VideoCapture(url)

    fps = input_video.get(cv2.CAP_PROP_FPS)
    #    multiplier = fps * 1
    try:
        while input_video.isOpened():
            frameId = int(round(input_video.get(1)))
            ret, image = input_video.read()
            image = rescale_frame(image, percent=75)
            image_original = image

            #            if (frameId % math.floor(multiplier) == 0):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = dt.face_detector.detect_faces(image)
            print("Found {} faces! ".format(len(faces)))
            for face in faces:
                embedding = extc.face_embeddings.extract_face_embeddings(image, face, shape_predictor, face_recognizer)
                label = recognize_face(embedding, embeddings, labels)
                (x1, y1, x2, y2) = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(image_original, (x1, y1), (x2, y2), (255, 120, 120), 2, cv2.LINE_AA)
                cv2.putText(image_original, label[0], (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 1)

            #            cv2.imshow("Image", image)
            cv2.imshow("Face Recognition using Dlib", image_original)
            if not ret:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        input_video.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(e)
        input_video.release()
        cv2.destroyAllWindows()
