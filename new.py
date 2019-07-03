import cv2
import numpy as np
import copy
import math
import pdb
import collections
#from appscript import app

# Environment:
# OS    : Mac OS EL Capitan
# python: 3.5
# opencv: 2.4.13

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))

def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx ** 2 + dy ** 2) ** .5
def circle_intersect(ca, cb):
    d = dist(ca['center'], cb['center'])
    r = ca['radius'] + cb['radius']
    return d < r
def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    fingers = []
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
                    fingers.append(far)
                
            return True, cnt, fingers
    return False, 0, []


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)


while camera.isOpened():
    
    
    ret, frame = camera.read()
   
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)


        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        #pdb.set_trace()
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull_r = cv2.convexHull(res, returnPoints=False)
            defects = cv2.convexityDefects(res, hull_r)
            drawing = np.zeros(img.shape, np.uint8)
            hull = cv2.convexHull(res)
            cv2.drawContours(drawing, [res], 0, (0, 255, 255), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
 
            min_val = 0
            
            min_index = []
            for i in range(hull.shape[0]):
                x, y = hull[i, 0]
                if y >= min_val:
                    min_val = y

            x_val = []
            for j in range(hull.shape[0]):
                x, y = hull[j, 0]
                if y == min_val:
                    x_val.append(x)
            avg_x = sum(x_val) / len(x_val)
            palm = [int(avg_x), min_val]
            cv2.circle(drawing, tuple(palm), 8, [100, 100, 100], 5)
            isFinishCal, cnt, fingers = calculateFingers(res,drawing)
            if len(fingers) > 0:
                for i in range(len(fingers)):
                    cv2.line(drawing, tuple(palm), fingers[i], [100, 100, 100], 2)
            resource_table = {}
            
            if defects is None:
                cv2.putText(drawing, 'no defects', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                continue
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                far = res[f][0]
                dis = dist(palm, far)
                resource_table[dis] = i
            #pdb.set_trace()
            items = resource_table.items()
            sort_items = sorted(items)
            resource = []
            for i in range(len(sort_items)):
                if i < 4:
                    resource.append(sort_items[i][1])

            for pt in resource:
                s, e, f, d = defects[pt, 0]
                far = res[f][0]
                cv2.circle(drawing, tuple(far), 8, [100, 100, 100], 5)

            for index in range(0, len(resource) - 1):
                for jndex in range(0, len(resource) - 1):
                    if index != jndex:
                        s, e, f, d = defects[resource[index], 0]
                        s2, e2, f2, d2 = defects[resource[jndex], 0]
                        far = res[f][0]
                        far2 = res[f2][0]
                        y = abs(far[1] - far2[1])
                        if y < 20:
                            cv2.line(drawing, tuple(far), tuple(far2), [211, 255, 0], 1)
                            

            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 2:
                    print (cnt)
                    #app('System Events').keystroke(' ')  # simulate pressing blank space


                    

        cv2.imshow('output', drawing)

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')
