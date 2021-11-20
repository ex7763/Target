from collections import deque

import numpy as np
import imutils
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear
import cv2

from shapely.geometry import Point
from sklearn.cluster import KMeans

def find_target(img):
    # find target
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,168], dtype=np.uint8)
    upper_white = np.array([172,111,255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
    white_mask = cv2.erode(white_mask, None, iterations=10)
    white_mask = cv2.dilate(white_mask, None, iterations=5)

    cnts = cv2.findContours(white_mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(white_mask, (x,y), (x+w, y+h), (255,255,0), 1)

    restricted = cv2.rectangle(white_mask, (x,y), (x+w, y+h), (255,255,0), -1)
    return restricted

def find_target_center(img):
    center_dict = {}
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red_0 = np.array([0, 70, 0])
    upper_red_0 = np.array([5, 255, 255])
    lower_red_1 = np.array([175, 70, 0])
    upper_red_1 = np.array([180, 255, 255])

    red_mask0 = cv2.inRange(hsv_img, lower_red_0, upper_red_0)
    red_mask1 = cv2.inRange(hsv_img, lower_red_1, upper_red_1)
    red_mask = cv2.bitwise_or(red_mask0, red_mask1)

    cnts = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        print(len(cnts))
        for cnt in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            print(x, y, radius)
            if radius > 100:
                M = cv2.moments(cnt)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                cv2.circle(img, (int(x), int(y)), int(radius),
                        (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)

    return img, center_dict


def find_arrow(img):
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    pos = None
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        if w > 50 or h > 50:
            pos = None
        else:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 1)
            pos = (x,y,w,h)
            print('\t',pos)
    return img, pos


if __name__ == '__main__':
    source = "/Volumes/Auxiliary/drive-download-20211114T071134Z-001/V_20211113_153244_vHDR_On.mp4"
    #source = "/Volumes/Auxiliary/data01.mp4"
    print(source)

    cap = VideoGear(source=source, stabilize = True).start()

    #cap = cv2.VideoCapture(source)
    #fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    # int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    # int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) # Rotate video

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #video_out = cv2.VideoWriter('output.avi', fourcc, fps, size)
    video_out = cv2.VideoWriter('output.avi', fourcc, 30.0, (int(1080), int(1920)))

    # MOG2
    fgbg = cv2.createBackgroundSubtractorMOG2()

    arrow_candidate = deque(maxlen=10)
    arrow_checked = []

    freeze_count = 0
    freeze_length = 60 * 10
    try:
        while(1):
            frame = cap.read()
            frame=cv2.transpose(frame)
            frame=cv2.flip(frame, 0)

            target_mask = find_target(frame)

            masked_frame = cv2.bitwise_and(frame, frame, mask=target_mask)

            fgmask = fgbg.apply(masked_frame)
            blurred = cv2.GaussianBlur(fgmask, (13, 13), 0)
            ret,thresh1 = cv2.threshold(blurred,250,255,cv2.THRESH_BINARY)
            arrow, arrow_pos = find_arrow(thresh1)

            target_center_img, target_center = find_target_center(masked_frame)
            if arrow_pos is not None:
                p = Point(int(arrow_pos[0]), int(arrow_pos[1]))
                # for c in arrow_candidate_lst:
                    # if p.distance(c) < 30:
                        # target_center_img = cv2.circle(target_center_img, (int(arrow_pos[0]), int(arrow_pos[1])), 5, (255, 0, 255), -1)
                        # break
                # else:
                    # arrow_candidate_lst.append(Point(int(arrow_pos[0]), int(arrow_pos[1])))
                #arrow_candidate_lst.append(Point(int(arrow_pos[0]), int(arrow_pos[1])))

                arrow_candidate.append([int(arrow_pos[0]), int(arrow_pos[1])])


            if len(arrow_candidate) > 2:
                kmeans = KMeans(n_clusters=2).fit(arrow_candidate)
                print(kmeans.cluster_centers_)
                c = kmeans.cluster_centers_[0].astype(int)
                target_center_img = cv2.circle(target_center_img, (c[0],c[1]), 15, (255, 0, 255), -1)
                c = kmeans.cluster_centers_[1].astype(int)
                target_center_img = cv2.circle(target_center_img, (c[0],c[1]), 15, (255, 0, 255), -1)

                print("score:",kmeans.inertia_)

                if kmeans.inertia_ < 1000.0 and freeze_count == 0:
                    x = np.argmax(np.bincount(kmeans.labels_))
                    arrow_checked.append([kmeans.cluster_centers_[x].astype(int)[0],
                                          kmeans.cluster_centers_[x].astype(int)[1]])
                    freeze_count = freeze_length

            for i in list(arrow_candidate):
                target_center_img = cv2.circle(target_center_img, (i[0],i[1]), 10, (255, 0, 0), 2)

            for i in list(arrow_checked):
                target_center_img = cv2.circle(target_center_img, (i[0],i[1]), 10, (0, 0, 255), -1)
            print(f"checked arrow: {len(arrow_checked)}")

            freeze_count = max(0, freeze_count-1)

            #cv2.imshow('fgmask',frame)
            #cv2.imshow('frame',fgmask)
            #cv2.imshow('thresh',arrow)

            #cv2.imshow('center', target_center_img)
            video_out.write(target_center_img)

            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
                # break

    finally:
        video_out.release()
        cap.release()
        cv2.destroyAllWindows()

