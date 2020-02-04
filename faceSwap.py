import cv2
import dlib
import numpy as np

#Get triangle index
def extract_index_nparray(nparray):
    index=None
    for num in nparray[0]:
        index=num
        break
    return index;

#finds face/s inside of the image
def findFace(image,faceDetector):
    faces = detector(image)
    return faces
    #print(faces)

#finding face landmarks on the given face inside of the given image    
def findFaceLandmakrs(image,face,landmarkPredictor):
    landmarks = landmarkPredictor(image,face)
    landmarks_points = []
    for n in range(0,68):
        x=landmarks.part(n).x
        y=landmarks.part(n).y
        landmarks_points.append((x,y))
        cv2.circle(image,(x,y),1,(255,0,0),-1)
    points = np.array(landmarks_points, np.int32) 
    convexhull=cv2.convexHull(points)
    #points = np.array(landmarks_points,np.int32)
    return landmarks_points,convexhull

#creating convex face mask based landmark points
def createConvexFaceMask(image,image_zero,faceLandmarks):
    
    points = np.array(faceLandmarks,np.int32)
    #convexhull "cuts" connects some angles so there is no angle bigger than 180*
    face_convexhull = cv2.convexHull(points)
    #cv2.polylines(image1,[face1_convexhull],True,(255,0,0),1)
    #popunjavamo masku sa bijelom bojom(gdje bi lice trebalo bit)
    #read about this function...
    cv2.fillConvexPoly(image_zero,face_convexhull,255)
    #bitwise pogledat objasnjenje funkcije
    image_face = cv2.bitwise_and(image,image,mask=image_zero)
    #cv2.imshow("image_face", image_face)
    return face_convexhull

#Divide face into triangles READ READ READ...
def createFaceTriangles(image,image_convexhull,faceLandmarks):
    points = np.array(faceLandmarks,np.int32)
    rect = cv2.boundingRect(image_convexhull)
    #(x, y, w, h)=rect
    #cv2.rectangle(image_face,(x,y),(x+w, y+h),(0,255,0))
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(faceLandmarks)
    triangles = subdiv.getTriangleList()
    triangles=np.array(triangles, dtype=np.int32)

    triangles_indexes = []
    for t in triangles:
        #print("Triangle:", t)
        pt1=(t[0],t[1])
        pt2=(t[2],t[3])
        pt3=(t[4],t[5])

        #Drawing triangles on sent image
        #cv2.line(image,pt1,pt2,(0,255,0),1)
        #cv2.line(image,pt1,pt3,(0,255,0),1)
        #cv2.line(image,pt2,pt3,(0,255,0),1)

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle=[index_pt1, index_pt2, index_pt3]
            triangles_indexes.append(triangle)
            #print(indexes_triangles)
    return triangles_indexes

#Cropping triangle from image......
def cropTriangle(image,triangle_index,faceLandmarks):
    tr1_pt1 = faceLandmarks[triangle_index[0]]
    tr1_pt2 = faceLandmarks[triangle_index[1]]
    tr1_pt3 = faceLandmarks[triangle_index[2]]
    tr_points = [tr1_pt1,tr1_pt2,tr1_pt3]
    triangle=np.array([tr1_pt1,tr1_pt2,tr1_pt3],np.int32)
    rect = cv2.boundingRect(triangle)
    (x,y,w,h)=rect
    #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    points = np.array([[tr1_pt1[0]-x,tr1_pt1[1]-y],
                       [tr1_pt2[0]-x,tr1_pt2[1]-y],
                       [tr1_pt3[0]-x,tr1_pt3[1]-y]],np.int32)
    #cropping triangle inside of rectangle
    cropped_rectangle = image[y:y+h, x:x+w]
    cropped_rectangle_mask = np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(cropped_rectangle_mask,points,255)
    cropped_triangle = cv2.bitwise_and(cropped_rectangle,cropped_rectangle,mask=cropped_rectangle_mask)
    #cv2.imshow("cropped_rectangle", cropped_rectangle)
    #v2.imshow("cropped_rectangle_mask", cropped_rectangle_mask)
    #cv2.imshow("cropped_triangle", cropped_triangle)
    #print(rect)
    return points, rect, cropped_triangle,triangle,tr_points





#image paths.
image1Path = "face_detect_test5.jpg"
image2Path = "face_detect_test6.jpg"
#create dlib detector
detector = dlib.get_frontal_face_detector();
#create predictor for landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#reading images from file
image1 = cv2.imread(image1Path)
image2 = cv2.imread(image2Path)
image3 = np.zeros_like(image2)
image4 = np.zeros_like(image1)
#converting images to gray
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#creating masks
image1_mask = np.zeros_like(image1_gray)
image2_mask = np.zeros_like(image2_gray)
#new face
height, width, channels = image2.shape
img2_new_face = np.zeros((height, width, channels), np.uint8)

#finds face/s inside of the images(coordinates of rectangles)
faces1 = findFace(image1_gray,detector)
faces2 = findFace(image2_gray,detector)
face1 = faces1[0]
face2 = faces2[0]

#draw rectangle on face....
#x1 = face1.left()
#y1 = face1.top()
#x2 = face1.right()
#y2 = face1.bottom()
# Draw a rectangle around the faces
#cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 255, 0), 2)
#cv2.imshow("image11", image1)


#print(face1, face2)
#finding face landmarks on the given face inside of the given image 
face1Landmarks,convexHull1 = findFaceLandmakrs(image1_gray,face1,predictor)
face2Landmarks,convexHull2 = findFaceLandmakrs(image2_gray,face2,predictor)
#print(face1Landmarks, face2Landmarks)
#creating convex face mask based landmark points
image1_convexhull = createConvexFaceMask(image1,image1_mask,face1Landmarks)
image2_convexhull = createConvexFaceMask(image2,image2_mask,face2Landmarks)
#divideing face into triangle
image1_triangles_indexes = createFaceTriangles(image1,image1_convexhull,face1Landmarks)
image2_triangles_indexes = createFaceTriangles(image2,image2_convexhull,face2Landmarks)


#cv2.imshow("image3", image3)
#cv2.imshow("22",image2)

lines_space_mask = np.zeros_like(image1_gray)
lines_space_new_face = np.zeros_like(image2)



#Cropping triangles
for triangle_index in image1_triangles_indexes:
    points1, rect1, cropped_triangle1, triangle1, tr_points1 = cropTriangle(image1,triangle_index,face1Landmarks)
    points2, rect2, cropped_triangle2, triangle2, tr_points2 = cropTriangle(image2,triangle_index,face2Landmarks)

    (x1, y1, w1, h1) = rect1
    (x2, y2, w2, h2) = rect2
    #print(rect2)
    #cv2.imshow("cropped_triangle1", cropped_triangle1)
    #cv2.imshow("cropped_triangle2", cropped_triangle2)


    #Lines space
    cv2.line(lines_space_mask, tr_points1[0], tr_points1[1], 255)
    cv2.line(lines_space_mask, tr_points1[1], tr_points1[2], 255)
    cv2.line(lines_space_mask, tr_points1[0], tr_points1[2], 255)
    lines_space = cv2.bitwise_and(image1, image1, mask=lines_space_mask)
    #cv2.imshow("lines_space",lines_space)
    
    #ss
    cropped_tr2_mask = np.zeros((h2, w2), np.uint8)
    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)


    #Warp triangles
    points1_float = np.float32(points1)
    points2_float = np.float32(points2)
    M = cv2.getAffineTransform(points1_float, points2_float)
    warped_triangle = cv2.warpAffine(cropped_triangle1, M, (w2, h2))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

    # Reconstructing destination face
    img2_new_face_rect_area = img2_new_face[y2: y2 + h2, x2: x2 + w2]
    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)


    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
    img2_new_face[y2: y2 + h2, x2: x2 + w2] = img2_new_face_rect_area



img2_new_face = cv2.medianBlur(img2_new_face,3)

cv2.imshow("img2_new_face", img2_new_face)
cv2.waitKey(0)

# Face swapped (putting 1st face into 2nd face)
img2_face_mask = np.zeros_like(image2_gray)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexHull2, 255)
img2_face_mask = cv2.bitwise_not(img2_head_mask)


img2_head_noface = cv2.bitwise_and(image2, image2, mask=img2_face_mask)
result = cv2.add(img2_head_noface, img2_new_face)





(x, y, w, h) = cv2.boundingRect(convexHull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))



seamlessclone = cv2.seamlessClone(result, image2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)


cv2.imshow("seamlessclone", seamlessclone)
cv2.waitKey(0)





print("Hello world!")
#cv2.imshow("image1", image1)
#cv2.imshow("image2", image1)
#cv2.imshow("image1_mask", image1_mask)
#cv2.imshow("image1_face", image1_face)
#cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()