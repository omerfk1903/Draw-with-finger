import cv2 as cv # görüntü işleme
import mediapipe as mdp # google parmak kontrol
import numpy as np
from time import sleep # bekleme

# Parmakların konumlarını bulabilmek için x  ve y konum değerini pencere genişliği ile çarpımıyla bulunuyor.

control = True
Camera = cv.VideoCapture(0) # Bilgisayar kamerasına bağlanıyor
ret,Camera_Video = Camera.read(()) # görüntü okunuyor

# Boyut bilgisi
Frame_Size = Camera_Video.shape[:2]
Frame_Weight = Frame_Size[1]
Frame_Height = Frame_Size[0]

# Siyah ekran oluşturuldu
Black_img = np.zeros([Frame_Height,Frame_Weight],np.uint8)

# Elin bulunabilmesi için temel bilgiler ekleniyor
mphands = mdp.solutions.hands
mphands_draw = mdp.solutions.drawing_utils # cizim bilgileri ekleniyor
hands = mphands.Hands() # el bulma fonksiyonu başlatılıyor

print(f" Weight : {str(Frame_Weight)} | Height : {str(Frame_Height)} ")
print(" Görüntü geliyormu : {0} ".format(str(ret)))

sleep(1)# 1 saniye bekleme

while control :

    ret,Camera_Video = Camera.read(())
    Camera_Video_RGB = cv.cvtColor(Camera_Video,cv.COLOR_BGR2RGB) # RGB çeviriliyor

    HLMS = hands.process(Camera_Video_RGB) # Gelen görüntü işlenmeye başlıyor
    HLMS_OUT = HLMS.multi_hand_landmarks

    if not HLMS_OUT : print(" Hands not found ")
    else :
        for landmarks in HLMS_OUT :
            mphands_draw.draw_landmarks(Camera_Video,landmarks,mphands.HAND_CONNECTIONS)
            hands_count = len(HLMS_OUT) # Görüntüdeki el sayısı

            for finger_count,finger_location in enumerate(landmarks.landmark) :
                position_X,position_Y = int(finger_location.x * Frame_Weight),int(finger_location.y * Frame_Height)

                if finger_count == 1 :
                    position_X = Frame_Weight - (2 * position_X)

                    if position_X < 0 :
                        position_X = position_X * -1
                    
                    Black_img = cv.line(Black_img,(position_X,position_Y),(position_X+1,position_Y+1),(255,0,0),5)

    cv.imshow(" Kamera ",Camera_Video)
    cv.imshow(" Resim defteri ",Black_img)

    if (cv.waitKey(1) & 0xFF == ord("q")):
        control = False
        break

# görüntü yenileniyor
Black_img.release()
Camera_Video.release()
cv.destroyAllWindows()