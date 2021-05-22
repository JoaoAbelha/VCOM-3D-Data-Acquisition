import urllib.request
import cv2
import numpy as np
import time

# ir À app
# ultimo ponto onde diz start server
#copiar para a prox linha o que diz la
url='http://192.168.1.46:8080/shot.jpg'

# saves several photos in the toSavePath folder
# q -> quit
# space -> take a photo
def takeShots(toSavePath):
    counter = 1
    while True:
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
        img = cv2.imdecode(imgNp,-1)
        cv2.imshow('IPWebcam',img)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break
        elif k & 0xFF == 32:
            img_name = "frame_{}.png".format(counter)
            cv2.imwrite(toSavePath + "/"+  img_name, img)
            print("{} saved!".format(img_name))
            counter+=1

#takeShots('./imgs')



