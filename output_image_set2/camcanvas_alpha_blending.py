import numpy as np
import cv2
'''img_data = []
image1 = cv2.imread('11.jpg' ,1)
image2 = cv2.imread('12.jpg', 1)
image3 = cv2.imread('13.jpg', 1)
image4 = cv2.imread('14.jpg', 1)
image5 = cv2.imread('15.jpg', 1)

canvas1 = np.zeros([3*image1.shape[0],4*image1.shape[0],3],dtype ='uint8')
img_data.append(image2)
img_data.append(image3)
img_data.append(image4)
img_data.append(image5)
#print(np.shape(img_data))
#print(img_data)
#print(image1)'''
x = 300.0
y = 150.0
#a = np.array([[0,0,x],[0,0,y],[0,0,0]])
h = np.array([[1.0,0.0,x],[0.0,1.0,y],[0.0,0.0,1.0]])

#canvas[y:y+image1.shape[0],x:x+image1.shape[1]] = image1
#cv2.imwrite('c.jpg',canvas)
def homo(Image1,Image2,ta,i):
	img1 = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY)
	#cv2.imshow('img2',img1)

	orb = cv2.ORB_create()

	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	matches = bf.match(des1,des2)

	#Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	matches=matches[:20]
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:],None,flags=2)
	#cv2.imshow('matching',img3)
	dst_pts = np.float32([ kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
	src_pts = np.float32([ kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,1.0)
	#print M	
	if(i==0):
		M1 = np.matmul(h.astype(float),M.astype(float))
	else:
		M1 = np.matmul(M.astype(float),h.astype(float))	
	return (M,M1)


#print h
global canvas
def stitch(image3,m):
	global canvas
	dst = cv2.warpPerspective(image3,m,(canvas.shape[1],canvas.shape[0]),flags=cv2.INTER_NEAREST)
	#ind1 = np.where(canvas==0)
	ind_common = np.where((canvas!=0) & (dst!=0))
	canvas[ind_common] = canvas[ind_common]*0.1 + dst[ind_common]*0.9
	ind1 = np.where((canvas==0) & (dst!=0))	
	canvas[ind1] = dst[ind1]
		
			
	#canvas[ind1] = dst[ind1]
cap = cv2.VideoCapture(0)
_,frame = cap.read()
cap.release()
img1 = frame
cv2.imwrite('image' + str(1) + '.jpg',img1)
canvas1 = np.zeros([3*img1.shape[0],4*img1.shape[0],3],dtype ='uint8')
canvas = cv2.warpPerspective(img1,h,(canvas1.shape[1],canvas1.shape[0]),flags=cv2.INTER_NEAREST)
n=0
cap = cv2.VideoCapture(0)
cv2.waitKey(3000)
_,frame = cap.read()
img2 = frame
cv2.imwrite('image' + str(2) + '.jpg',img2)
cap.release()
while(n<5):
	_ , m  = homo(img1,img2,h,0)
	h = m	
	stitch(img2,m)
	cap = cv2.VideoCapture(0)
	cv2.waitKey(3000)
	_,frame = cap.read()
	img3 = frame
	cap.release()
	img1 = img2
	img2 = img3
	n = n+1
	cv2.imwrite('image' + str(n+2) + '.jpg',img3)
cv2.imwrite('canvas_alpha_blending.jpg',canvas)

cv2.destroyAllWindows()


