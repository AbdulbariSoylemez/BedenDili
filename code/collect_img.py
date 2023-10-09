## Veri setimizi oluştuturuyoruz
import os # Dosya ve dizin işlemleri için kulnılır 

import cv2 

DATA_DIR = "data" # verilerin kaydedileceği yer 
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 27  # kaç farklı değişken 
dataset_size = 80 # veri sayısı

cap = cv2.VideoCapture(0)
for j in range(number_of_classes): # Her bir veri sınıfı için dongü olşurduk 
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j))) 

    print('Collecting data for class {}'.format(j)) # hangi sınıf için veri tolandığını gösteriyor 

    done = False
    while True:
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'): # "q" tuşuna basılana kadar bu döngü devam eder.
            break

    counter = 0
    while counter < dataset_size: # sayaç 80 olduğunda döngüye girilmiyecek 
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame) # verimizi yazdırıyoruz 

        counter += 1 

cap.release()
cv2.destroyAllWindows()






