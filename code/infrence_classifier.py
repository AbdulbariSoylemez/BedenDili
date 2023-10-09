import numpy as np
import cv2 
import mediapipe as mp
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

model_dick = pickle.load(open("model.p", "rb"))
model = model_dick["model"]
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

video = cv2.VideoCapture(0)

while True:
    data_aux = []  # Her döngü başında temiz bir liste oluşturun
    x_=[]
    y_=[]

    ret, frame = video.read()
    H, W, _ = frame.shape
    
    if ret == 0:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.extend([x, y])  # El işaretlerini data_aux listesine ekleyin
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Şimdi tahmin yapabilirsiniz
        prediction = model.predict([np.array(data_aux)])  # data_aux'ı NumPy dizisine dönüştürün

        predicted_character = labels_dict[int(prediction[0])]


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,  cv2.LINE_AA)

        print("Tahmin: ", predicted_character)

    frame = cv2.flip(frame, 1)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()


