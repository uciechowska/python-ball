# from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

#poczatek czasu do kalibracji
calstart = time.time()

posX = []
posY = []

showRGB = True

while(True):
    # Capture frame-by-frame
    ret, frameRGB = cap.read()

    #
    # konwersja na HSV aby uzyskac wartosci barwy (Hue), nasycenia i jasnosci do kalibracji koloru kulki
    frameHSV = cv2.cvtColor(frameRGB, cv2.COLOR_BGR2HSV)
    #rozmiar obrazu
    h,w,chan = frameRGB.shape
    h2 = h/2
    w2 = w/2

    #kalibracja HSV
    if time.time() - calstart < 10:
        side = 10
        #dodajemy krzyzyk do klatki
        cv2.line(frameRGB,(w2-side,h2-side),(w2+side,h2-side),(0,255,0),2)
        cv2.line(frameRGB,(w2-side,h2+side),(w2+side,h2+side),(0,255,0),2)
        cv2.line(frameRGB,(w2-side,h2-side),(w2-side,h2+side),(0,255,0),2)
        cv2.line(frameRGB,(w2+side,h2-side),(w2+side,h2+side),(0,255,0),2)

        #pobieram srodek ekranu (6x6 pikseli) aby odczytac srednie wartosci HSV kulki
        roi = np.array(frameHSV[(h2-side):(h2+side),(w2-side):(w2+side)])
        #liczymy potrzebne srednie
        meanH = np.mean(roi[:,:,0])
        meanS = np.mean(roi[:,:,1])
        meanV = np.mean(roi[:,:,2])

        frameToShow = frameRGB
    else:
        # print (meanH, meanS, meanV)
        # ustalamy zakres HSV
        mask = cv2.inRange(frameHSV, (meanH - 30,meanS - 30, meanV - 20), (meanH + 10,meanS + 20, meanV + 20))
         # Bitwise-AND mask and original image
        frameMasked = cv2.bitwise_and(frameRGB, frameRGB, mask = mask)

        if showRGB:
            frameToShow = frameRGB
        else:
            frameToShow = frameMasked
        #liczymy niezerowe pozycje x i y, czyli w ktorych jest kulka
        indY,indX = np.nonzero(mask)
        #jezeli znalazl dostatecznie duzo pikseli kulki
        if len(indX) > 1000:
            #srednia czyli srodek ciezkosci
            meanX = sum(indX)/len(indX)
            meanY = sum(indY)/len(indY)

            posX.append(meanX)
            posY.append(meanY)

            #albo dodajemy krzyzyk do klatki
            #cv2.line(frameToShow,(meanX-10,meanY),(meanX+10,meanY),(0,255,0),5)
            #cv2.line(frameToShow,(meanX,meanY-10),(meanX,meanY+10),(0,255,0),5)

            #albo  wyswietlamy cala trajektorie
        frameToShow[np.array(posY)-1,np.array(posX)] = (0,255,0)
        frameToShow[np.array(posY)+1,np.array(posX)] = (0,255,0)
        frameToShow[np.array(posY),np.array(posX)-1] = (0,255,0)
        frameToShow[np.array(posY),np.array(posX)+1] = (0,255,0)

            #TODO: tutaj dodac rozpoznawanie literki M z trajektorii (positions)
            ###################################################
    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        #cv2.imwrite('klatka.png',frame)
        break
    if k & 0xFF == ord('a'):
        showRGB = not showRGB

    #wyswietlamy klatke3
    cv2.imshow('',frameToShow)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
