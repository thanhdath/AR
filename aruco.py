# import numpy as np
# import cv2
# import cv2.aruco as aruco


# '''
#     drawMarker(...)
#         drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
# '''

# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
# print(aruco_dict)
# # second parameter is id number
# # last parameter is total image size
# img = aruco.drawMarker(aruco_dict, 2, 700)
# cv2.imwrite("test_marker.jpg", img)

# cv2.imshow('frame',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import sys;sys.exit()



import numpy as np
import cv2
import cv2.aruco as aruco


cap = cv2.VideoCapture(0)
import imgaug.augmenters as iaa
aug = iaa.Sequential([
    iaa.PerspectiveTransform(scale=(0.0, 0.1))
])
import time

# camera_matrix = np.array([[800, 0, 320], [0, 600, 240], [0, 0, 1]])
# dist_coeffs = np.array([1,1,1,1])

camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeff.npy')

while(True):
    # Capture frame-by-frame
    # ret, frame = cap.read()
    frame = cv2.imread('surface.png')
    frame = aug.augment_image(frame)
    time.sleep(0.2)

    #print(frame.shape) #480x640
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    parameters =  aruco.DetectorParameters_create()

    #print(parameters)

    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
        #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # print(corners[0].shape)

    # print(rejectedImgPoints)

    #It's working.
    # my problem was that the cellphone put black all around it. The alrogithm
    # depends very much upon finding rectangular black blobs

    gray = aruco.drawDetectedMarkers(frame, corners, ids)
    # cv2.polylines(frame, corners[0].astype(np.int32), True, color=(0,255,0))
    # cv2.polylines(frame, [x.reshape(-1,2).astype(np.int32) for x in rejectedImgPoints], True, color=(0,0,255))
    rvec, tvec, cc = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
    # rvec: rotation vector, tvec: translation vector
    print(rvec, tvec)
    print(cc)
    aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    # pt = np.array([0,0,0])
    # # import pdb; pdb.set_trace()
    # pt = pt.dot(rvec)
    # pt = pt.dot(tvec)
    # pt = 

    #print(rejectedImgPoints)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.waitKey(0)
    break

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()