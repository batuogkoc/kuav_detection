import numpy as np
import cv2 as cv
import glob
import os
pattern_height = 7
pattern_width = 4
board_size = 10
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cap = cv.VideoCapture(0)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
photo_counter=1
img_directory = "calibration_images"
while True:
    ret,frame = cap.read()
    # cv.imshow("Video",frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_out = np.copy(frame)
    ret, corners = cv.findChessboardCorners(gray, (pattern_width,pattern_height), None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        cv.drawChessboardCorners(frame_out, (pattern_width,pattern_height), corners2, ret)
    cv.imshow('img', frame_out)
    key = cv.waitKey(10)
    if (key & 0xFF == ord("p")) and ret:
        cv.imwrite(os.path.join(img_directory, "resim"+str(photo_counter)+".jpg"),frame)
        photo_counter+=1
    elif key & 0xFF == ord("q"):
        break
cap.release()


objp = np.zeros((pattern_height*pattern_width,3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_width,0:pattern_height].T.reshape(-1,2)*board_size
objpoints = []
imgpoints = []
images = glob.glob(os.path.join(img_directory, '*.jpg'))
calibrated=1
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (pattern_width,pattern_height), None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        cv.drawChessboardCorners(img, (pattern_width,pattern_height), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Ret Results:",ret)
print("Mtx Results:",mtx)
print("Dist Results:",dist)
# print("Rvecs Results:",rvecs)
# print("Tvecs Results:",tvecs)
cv.destroyAllWindows()
