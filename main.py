import cv2
import numpy as np
import mediapipe as mp
import datetime
import tensorflow as tf
import pickle
from sklearn import neighbors
import pandas as pd
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
"""
gpus=tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu,True)
		logical_gpus=tf.config.experimental.list_logical_devices('GPU')

		print(len(gpus),len(logical_gpus))
	except RuntimeError as e:
		print(e)


model = tf.keras.models.load_model("cnn_model_keras2.h5")

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist
"""

df=pd.read_csv("sibi.csv")
model=neighbors.KNeighborsClassifier(n_neighbors=3,metric="euclidean",weights="distance")
X=df[['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4','x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8',
      'x9', 'y9', 'x10','y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 'x15','y15', 'x16', 'y16',
      'x17', 'y17', 'x18', 'y18', 'x19', 'y19', 'x20','y20']]
y=df['label']
model.fit(X,y)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(1)

with mp_hands.Hands(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5,
		max_num_hands=1) as hands:
	while cap.isOpened():
		success, image = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			# If loading a video, use 'break' instead of 'continue'.
			continue
		# Flip the image horizontally for a later selfie-view display, and convert
		# the BGR image to RGB.
		h,w,c=image.shape
		image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
		rawimage=image[:]
		rawimage = cv2.cvtColor(rawimage, cv2.COLOR_RGB2BGR)
		char_op="null"
		# To improve performance, optionally mark the image as not writeable to
		# pass by reference.
		image.flags.writeable = False
		results = hands.process(image)
		# Draw the hand annotations on the image.
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		image2=np.zeros((300,300,3),np.uint8)
		hasil=np.zeros((180,180,3),np.uint8)
		
		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
				x_max,y_max,x_min,y_min=0,0,w,h
				for lm in hand_landmarks.landmark:
					x,y=int(lm.x*w),int(lm.y*h)
					if x>x_max:
						x_max=x
					if x<x_min:
						x_min=x
					if y>y_max:
						y_max=y
					if y<y_min:
						y_min=y		
				#cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(0,255,0),2)
				x_mid,y_mid=int(x_min+((x_max-x_min)/2)),int(y_min+((y_max-y_min)/2))
				#x_mid,y_mid=int(hand_landmarks.landmark[0].x*w),int(hand_landmarks.landmark[0].y*h)
				panjang,lebar=x_max-x_min,y_max-y_min
				panjang=panjang+80
				lebar=lebar+80
				#cv2.circle(image,(x_mid,y_mid),2,(0,0,255),5)
				if(panjang>lebar):
					if(x_mid+int(panjang/2)<w and y_mid+int(panjang/2)<h and y_mid-int(panjang/2)>0 and x_mid-int(panjang/2)>0 and x_mid>0 and y_mid>0):
						cv2.rectangle(image,(x_mid-int(panjang/2),y_mid-int(panjang/2)),(x_mid+int(panjang/2),y_mid+int(panjang/2)),(0,0,255),2)
						image2=rawimage[y_mid-int(panjang/2):y_mid+int(panjang/2),x_mid-int(panjang/2):x_mid+int(panjang/2)]
				elif(lebar>panjang):
					if(x_mid+int(lebar/2)<w and y_mid+int(lebar/2)<h and y_mid-int(lebar/2)>0 and x_mid-int(lebar/2)>0 and x_mid>0 and y_mid>0):
						cv2.rectangle(image,(x_mid-int(lebar/2),y_mid-int(lebar/2)),(x_mid+int(lebar/2),y_mid+int(lebar/2)),(0,0,255),2)
						image2=rawimage[y_mid-int(lebar/2):y_mid+int(lebar/2),x_mid-int(lebar/2):x_mid+int(lebar/2)]
				else:
					image2=np.zeros((300,300,3),np.uint8)
				
				#if cv2.waitKey(5) & 0xFF == ord("c"):
				image2 = cv2.resize(image2, (300,300), interpolation = cv2.INTER_AREA)
				koordinat=[]
				for num,i in enumerate(hand_landmarks.landmark):
					if(panjang>lebar):
						# Koordinat berdasarkan scale gambar satuan skala
						x_landmark=((i.x*w)-(x_mid-int(panjang/2)))/panjang
						y_landmark=((i.y*h)-(y_mid-int(panjang/2)))/panjang
						# Koordinat berdasarkan box
						txt=str(num)+" x="+str(int(i.x*w)-(x_mid-int(panjang/2)))+",y="+str(int(i.y*h)-(y_mid-int(panjang/2)))
					if(lebar>panjang):
						# Koordinat berdasarkan scale gambar satuan skala
						x_landmark=((i.x*w)-(x_mid-int(lebar/2)))/lebar
						y_landmark=((i.y*h)-(y_mid-int(lebar/2)))/lebar
						# Koordinat berdasarkan box
						txt=str(num)+" x="+str(int(i.x*w)-(x_mid-int(lebar/2)))+",y="+str(int(i.y*h)-(y_mid-int(lebar/2)))
					cv2.putText(image,txt,(int(i.x*w),int(i.y*h)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
					
					# Koordinat berdasarkan scale gambar
					#cv2.circle(image2,(int(x_landmark*300),int(y_landmark*300)),1,(0,0,255),2)
					#cv2.putText(image2,str(int(x_landmark*300))+","+str(int(y_landmark*300)),(int(x_landmark*300),int(y_landmark*300)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
					#print(num,i.x*w,i.y*h)
					koordinat.append(num)
					koordinat.append(int(x_landmark*300))
					koordinat.append(int(y_landmark*300))
				
				if cv2.waitKey(5) & 0xFF == ord("c"):
					
					datestamp=datetime.datetime.now()
					filename=datestamp.strftime("%Y%m%d %H%M%S %f")
					print(koordinat)
					print(filename)
					with open("data/txt/"+filename+".txt","w") as output:
						output.write(str(koordinat))
					cv2.imwrite("data/rps/"+filename+".jpg",image2)
				
				"""
				#evilport method using image hist
				hist = get_hand_hist()
				imgHSV = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
				dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
				disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
				cv2.filter2D(dst,-1,disc,dst)
				blur = cv2.GaussianBlur(dst, (11,11), 0)
				blur = cv2.medianBlur(blur, 15)
				thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
				thresh = cv2.merge((thresh,thresh,thresh))
				thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
				thresh = thresh[0:300, 0:300]
				(openCV_ver,_,__) = cv2.__version__.split(".")
				if openCV_ver=='3':
					contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
				elif openCV_ver=='4':
					contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
				if len(contours) > 0:
					contour = max(contours, key = cv2.contourArea)
					#print(cv2.contourArea(contour))
					if cv2.contourArea(contour) > 10000:
						x1, y1, w1, h1 = cv2.boundingRect(contour)
						save_img = thresh[y1:y1+h1, x1:x1+w1]
						if w1 > h1:
							save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
						elif h1 > w1:
							save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
						img = cv2.resize(save_img, (50, 50))
						img = np.array(img, dtype=np.float32)
						img = np.reshape(img, (1, 50, 50, 1))
						processed = img
						pred_probab = model.predict(processed)[0]
						pred_class = list(pred_probab).index(max(pred_probab))
						#print(pred_probab)
						if max(pred_probab)*100 > 50:
							alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O",
							"P","Q","R","S","T","U","V","W","X","Y","Z",
							"0","1","2","3","4","5","6","7","8","9",
							"-","-","-","-","-","-","-","-",]
							text = alphabet[pred_class]
							cv2.putText(hasil,text, (30,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
				"""
				#predicted=model.predict([koordinat])
				#cv2.putText(hasil,predicted[0], (30,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
		cv2.imshow('Camera', image)
		cv2.imshow('Hands', image2)
		cv2.imshow('Predictions', hasil)		
		if cv2.waitKey(5) & 0xFF == 27:
			break
cap.release()