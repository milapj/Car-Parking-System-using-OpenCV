# import the necessary packages
import cv2
import numpy as np
import imutils
import argparse
import sys
import time
import pyrebase
import thread

start_time = time.time()

# setting up firebase parameters
config = {
    "apiKey": "AIzaSyDLu8IDF9n4icmovsT_uw-ZuZe83Xpry10",
    "authDomain": "parkinglot1-details.firebaseapp.com",
    "databaseURL": "https://parkinglot1-details.firebaseio.com",
    "projectId": "parkinglot1-details",
    "storageBucket": "parkinglot1-details.appspot.com",
    "messagingSenderId": "163381767838"
}

# setting up firebase nodes
firebase = pyrebase.initialize_app(config)
db = firebase.database()
db.child("Incoming").child("Medium")
db.child("Incoming").child("Large")
db.child("Outgoing").child("Medium")
db.child("Outgoing").child("Large")

# Set BGSMOG to an object 
fgbg = cv2.createBackgroundSubtractorMOG2()


def firebase(bus_count_left, bus_count_right, car_count_left, car_count_right):
    # Push Data into Firebase
    if bus_count_right > 0:
        data = {"Count": str(bus_count_right)}
        db.child("Incoming").child("Large").set(data)
    if car_count_right > 0:
        data = {"Count": str(car_count_right)}
        db.child("Incoming").child("Medium").set(data)
    if bus_count_left > 0:
        data = {"Count": str(bus_count_left)}
        db.child("Outgoing").child("Large").set(data)
    if car_count_left > 0:
        data = {"Count": str(car_count_left)}
        db.child("Outgoing").child("Medium").set(data)


def contourfinder(frame, erosion, car_count_left, car_count_right, bus_count_left, bus_count_right):
    # set up reference line
    cv2.line(frame, (0, 520), (1280, 520), (0, 0, 255), 2)

    # Finding Contours
    image, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    try:
        hierarchy = hierarchy[0]
    except:
        hierarchy = []

    for c in contours:
        # display box for contours
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        if cv2.contourArea(c) > 1000:
            area = cv2.contourArea(c)
            # Set a display for centroid
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            center = (cX, cY)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            # display contour
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Vehicle Counting
            if cY > 520 and cY < 529:

                if cX > 650:
                    if area > 10000:
                        bus_count_right = bus_count_right + 1
                        cv2.line(frame, (0, 520), (1280, 520), (255, 255, 255), 2)
                        break
                    else:
                        car_count_right = car_count_right + 1
                        cv2.line(frame, (0, 520), (1280, 520), (255, 255, 255), 2)
                        break

                else:
                    if area > 10000:
                        bus_count_left = bus_count_left + 1
                        cv2.line(frame, (0, 520), (1280, 520), (255, 255, 255), 2)
                        break
                    else:
                        car_count_left = car_count_left + 1
                        cv2.line(frame, (0, 520), (1280, 520), (255, 255, 255), 2)
                        break


                    # file.write(str(car_count))
                    # file.write(str(bus_count))

    return (car_count_left, car_count_right, bus_count_left, bus_count_right)


def imageProcessing(frame):
    # Background Subtraction
    fgmask = fgbg.apply(frame)
    # median blur
    blur = cv2.medianBlur(fgmask, 5)
    # Gaussian Blur
    gblur = cv2.GaussianBlur(blur, (45, 45), 0)
    # thresholding
    ret, thresh = cv2.threshold(gblur, 127, 255, 3)
    # dilation and erosion
    kernel = np.ones((6, 6), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=3)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    return erosion


def main():
    # Video source
    # if a video path was not supplied, grab the default vid
    cap = cv2.VideoCapture('test_vid1.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Exit if video not opened.
    if not cap.isOpened():
        print "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = cap.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()

    # Find Video stats
    fps = cap.get(cv2.CAP_PROP_FPS)
    # calculate real time fps on pi
    # add code here
    frame_rate = int(fps)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # initialize count variable
    car_count_left, car_count_right, bus_count_left, bus_count_right = 0, 0, 0, 0
    frame_count = 0
    # set up pts for 2lane ROI
    rY1, rY2 = 0, 700
    rX1, rX2, rX3 = 0, 650, 1280

    while (cap.isOpened()):
        (grabbed, frame) = cap.read()
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if not grabbed:
            break
        # image processing on the frame
        erosion = imageProcessing(frame)
        # increment frame counter by 1
        frame_count += 1
        # finding contours
        car_count_left, car_count_right, bus_count_left, bus_count_right = contourfinder(frame, erosion, car_count_left,
                                                                                         car_count_right,
                                                                                         bus_count_left,
                                                                                         bus_count_right)

        try:
            thread.start_new_thread(firebase, (bus_count_left, bus_count_right, car_count_left, car_count_right,))
        except:
            print ("Error: unable to start thread")
        # Display Video Stats
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Display Stats
        cv2.putText(frame, "Frame Count:", (20, 180), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(frame_count), (20, 220), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Total Frames:", (20, 260), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(total_frame), (20, 300), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str("Frame Rate:"), (20, 340), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # left vehicle count
        cv2.putText(frame, str(frame_rate), (20, 380), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str("Medium vehicle:"), (20, 420), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(car_count_left), (20, 460), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str("Large Vehicle:"), (20, 500), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(bus_count_left), (20, 540), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # right vehicle count
        cv2.putText(frame, str("Medium vehicle:"), (1000, 420), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(car_count_right), (1000, 460), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str("Large Vehicle:"), (1000, 500), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(bus_count_right), (1000, 540), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # Display ROI
        cv2.rectangle(frame, (rX1, rY1), (rX2, rY2), (0, 255, 0), 3)
        cv2.rectangle(frame, (rX2, rY1), (rX3, rY2), (0, 255, 0), 3)
        # display video
        # cv2.imshow('BG Sub',fgmask)
        # cv2.imshow('SPN removal',gblur)
        # cv2.imshow('Thresholding',thresh)
        # cv2.imshow('Dilation', dilation)
        # cv2.imshow('Erosion', erosion)
        cv2.imshow('Video Output', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    print("--- %s seconds ---" % (time.time() - start_time))