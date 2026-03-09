import numpy as np
import cv2
import sys
# Classification Bayésienne avec distribution gaussienne multidimensionnelle :
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
#Idem, avec indépendance par composantes (i.e. covariance diagonale)
from sklearn.naive_bayes import GaussianNB 

if len(sys.argv) != 2:
  print ("Usage :",sys.argv[0],"<Image_in>")
  sys.exit(2)
else:
  img_bgr=cv2.imread(sys.argv[1],-1)
(h_img,w_img,c_img) = img_bgr.shape
print("Dimension de l'image :",h_img,"lignes x",w_img,"colonnes x",c_img,"canaux")
print("Type de l'image :",img_bgr.dtype)

# this function allows to draw a sample rectangle 
# and to get its coordinates in the training image
roi_defined = False
def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True
		
# Cette fonction met à jour le modèle bayésien avec le
# batch de pixels (RoI) courant, et le label fourni
def update_GBModel(r,c,w,h,label):
	global img_bgr,clf,classes,data_features,data_labels,first
	roi_features = img_bgr[c:c+w, r:r+h]
	print("RoI_Features :",roi_features.shape)
	roi_labels = np.full((w, h), label)
	batch_features = np.reshape(roi_features,(-1,3))
	batch_labels = np.ravel(roi_labels)
	if first == 1:
		first = 0
		data_features = batch_features.copy()
		data_labels = batch_labels.copy()
	else:
		data_features = np.concatenate((data_features,batch_features),axis = 0)
		data_labels = np.concatenate((data_labels,batch_labels),axis = 0)	

clone = img_bgr.copy()
cv2.namedWindow("Training image")
cv2.setMouseCallback("Training image", define_ROI)
num_pos = 0
num_neg = 0
first = 1
# Choix du modèle
# Par défaut de prior, ceux-ci sont calculés par la proportion de données fournies
# Alternativement on peut fixer les priors par ex : priors = [0.3,0.7]
clf = QuadraticDiscriminantAnalysis(priors = None)
#clf = GaussianNB(priors = None)

classes = np.unique([-1, 1])
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("Training image",img_bgr)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(img_bgr, (r,c), (r+h,c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		img_bgr = clone.copy()
	# if the 'p' key is pressed, record a positive batch sample
	if key == ord("p"):
	    num_pos += 1
	    update_GBModel(r,c,w,h,1)
	    print("Batch positif n°",num_pos,"enregistré !")
	# if the 'n' key is pressed, record a positive batch sample
	if key == ord("n"):
	    num_neg += 1
	    update_GBModel(r,c,w,h,-1)
	    print("Batch négatif n°",num_neg,"enregistré !")    	
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break

# First fit the whole data to the model
print("Dimension des features :",data_features.shape)
print("Dimension des labels :",data_labels.shape)
clf.fit(data_features,data_labels)  

# Test the prediction on the whole image
img_bgr = clone.copy()
img_samples = np.reshape(img_bgr,(-1,3))
img_labels = clf.predict(img_samples)
img_result = np.reshape(img_labels,(h_img,w_img))
print("Type de l'image de label :",img_result.dtype)
img_result[np.where(img_result == 1)] = 255
img_result[np.where(img_result == -1)] = 0
img_result = img_result.astype(np.uint8)
cv2.imshow("Testing on train image",img_result)
cv2.waitKey(0)
