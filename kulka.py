# from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import numpy as np
import cv2
import time
import math


# liczy kat (0... pi) pomiedzy kolejnymi trzema punktami
def calcAngle(ax,ay,bx,by,cx,cy):
    abx = bx - ax
    aby = by - ay
    bcx = cx - bx
    bcy = cy - by

    s = math.sqrt((abx**2 + aby**2)*(bcx**2 + bcy**2))
    if s == 0:
        return 0
    else:
        return math.acos((abx*bcx+aby*bcy)/s)

cap = cv2.VideoCapture(0)

#poczatek czasu do kalibracji
calstart = time.time()

#poczatek czasu do wyswietlania statusu
alertstart = 0

#tablica z punktami
posX = []
posY = []
#punkty ale tylko te w pewnej odleglosci
posDistX = []
posDistY = []
#punkty w odleglosci i pod duzym katem
posAcuteX = []
posAcuteY = []


# rozne sposoby wyswietlania
showRGB = True
showLines = True
showPoints = 3 #all | dist | acute | acute5

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
    if time.time() - calstart < 5:
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

            # tutaj zbieramy wszystkie punkty do wyswietlenia
            posX.append(meanX)
            posY.append(meanY)

            #na poczatku mamy pusta tablice, wiec po prostu dodajemy kilka pierwszych punktow
            if len(posAcuteX) < 2:
                posDistX.append(meanX)
                posDistY.append(meanY)
                posAcuteX.append(meanX)
                posAcuteY.append(meanY)
            else:
                #jesli nowa pozycja jest daleko od poprzednich
                if (meanX - posDistX[-1])**2 + (meanY - posDistY[-1])**2 > 40**2:
                    #tutaj zbieramy te w odpowiedniej pozycji
                    posDistX.append(meanX)
                    posDistY.append(meanY)

                    #sprawdzamy czy nie jest (z grubsza) wspolliniowa z poprzednimi
                    # https://math.stackexchange.com/questions/361412/finding-the-angle-between-three-points
                    #jezeli nie jest to dopisujemy ja

                    #A tutaj tylko te ktore sa pod odpowiednim katem
                    a = calcAngle(posAcuteX[-2],posAcuteY[-2], posAcuteX[-1], posAcuteY[-1], meanX, meanY)
                    if a < math.pi/3: # tutaj 60 stopni w dopelnieniu daje 120
                        posAcuteX.pop()
                        posAcuteY.pop()

                    posAcuteX.append(meanX)
                    posAcuteY.append(meanY)


        #wyswietlanie punktow
        pointsX = posX
        pointsY = posY

        if showPoints == 1:
            pointsX = posDistX
            pointsY = posDistY

        if showPoints >= 2:
            pointsX = posAcuteX
            pointsY = posAcuteY

        if showPoints == 3:
            pointsX = pointsX[-5:]
            pointsY = pointsY[-5:]


        #wyswietlamy cala trajektorie albo tylko niektore punkty
        if showLines: #jako linie
            cv2.polylines(frameToShow, [np.array(zip(pointsX,pointsY))], False, (0,255,0))
        else: # albo punktu (kolka)
            for x,y in zip(pointsX,pointsY):
                cv2.circle(frameToShow, (x,y), 3, (0,255,0), -1)


    #rozpoznawanie literki przy uzyciu 5 punktow
    if len(posAcuteX) >= 5:
        angleleft = calcAngle(posAcuteX[-5],posAcuteY[-5], posAcuteX[-4], posAcuteY[-4], posAcuteX[-4], posAcuteY[-4] - 10)
        angleright = calcAngle(posAcuteX[-2],posAcuteY[-2], posAcuteX[-1], posAcuteY[-1], posAcuteX[-1], posAcuteY[-1] + 10)
        angle1 = calcAngle(posAcuteX[-5],posAcuteY[-5], posAcuteX[-4], posAcuteY[-4], posAcuteX[-3], posAcuteY[-3])
        angle2 = calcAngle(posAcuteX[-4],posAcuteY[-4], posAcuteX[-3], posAcuteY[-3], posAcuteX[-2], posAcuteY[-2])
        angle3 = calcAngle(posAcuteX[-3],posAcuteY[-3], posAcuteX[-2], posAcuteY[-2], posAcuteX[-1], posAcuteY[-1])
#        print angleleft / math.pi * 180, angleright / math.pi * 180, angle1 / math.pi * 180, angle2 / math.pi * 180, angle3 / math.pi * 180

        #angle left/right:
        #   pierwsza i ostatnia kreska mniej wiecej pionowa ( 20 stopni):
        #angle 1/3:
        #   katy zewnetrzne ostre (przynajmniej 60 stopni) ale nie za ostre (mniej niz 15 stopni)
        #angle 2:
        #   kat w samym srodku moze byc nawet rozwarty (ale co najwyzej 120 stopni) ale nie za ostry (mniej niz 15 stopni)
        if  angleleft < math.pi/9 and \
            angleright < math.pi/9 and \
            angle1 > math.pi*2/3 and angle1 < math.pi*11/12 and \
            angle3 > math.pi*2/3 and angle3 < math.pi*11/12 and \
            angle2 > math.pi/3 and angle2 < math.pi*11/12:
            alertstart = time.time()

    #reagujemy na klawiature
    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        #cv2.imwrite('klatka.png',frame)
        break

    if k & 0xFF == ord('a'):
        showRGB = not showRGB

    if k & 0xFF == ord('l'):
        showLines = not showLines

    if k & 0xFF == ord('p'):
        showPoints = (showPoints + 1) % 4

    if k & 0xFF == ord('r'):
        showPoints = 3
        showLines = True
        showRGB = True
        posX = []
        posY = []
        posDistX = []
        posDistY = []
        posAcuteX = []
        posAcuteY = []

    #wyswietlamy klatke3
    #odbicie lustrzane, aby latwiej sie "pisalo M"
    frameToShow = cv2.flip(frameToShow,1)
    #legenda
    cv2.putText(frameToShow, 'Q - wyjscie', (30,30), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,255,255), 2)
    cv2.putText(frameToShow, 'A, P, L - wyswietlanie', (30,60), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,255,255), 2)
    cv2.putText(frameToShow, 'R - reset', (30,90), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,255,255), 2)

    if time.time() - alertstart < 1:
        cv2.putText(frameToShow, 'JEST LITERKA', (30,120), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2)

    cv2.namedWindow('mojeokno',flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
#    cv2.moveWindow('mojeokno', 600,200)
    cv2.imshow('mojeokno',frameToShow)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
