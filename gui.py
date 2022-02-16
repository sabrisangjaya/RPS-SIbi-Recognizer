import cv2
import numpy as np
import mediapipe as mp
import datetime,time
import tensorflow as tf
import pickle
from sklearn import neighbors
import pandas as pd
import tkinter as tk
from PIL import Image,ImageTk
import random
import sys,os

bot_hands=["rock","paper","scissors"]
bot_img={"paper":"data/rps/paper.jpg"
,"rock":"data/rps/rock.jpg"
,"scissors":"data/rps/scissors.jpg"}

waktu=None

bot_choice=None
player_choice=None
player_image=np.zeros((300,300,3),np.uint8)	

df=pd.read_csv("rps.csv")
model=neighbors.KNeighborsClassifier(n_neighbors=3,metric="euclidean",weights="distance")
X=df[['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4','x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8',
      'x9', 'y9', 'x10','y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 'x15','y15', 'x16', 'y16',
      'x17', 'y17', 'x18', 'y18', 'x19', 'y19', 'x20','y20']]
y=df['label']
model.fit(X,y)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def restart_program():
	python=sys.executable
	os.execl(python,python,* sys.argv)


cap = cv2.VideoCapture(0)

window=tk.Tk()
window.wm_title("RPS Recognizer")
window.config(background="#FFFFFF")
window.geometry('+%d+%d'%(0,0))
tk.Label(window,text=" ").grid(row=4,column=1)
tk.Label(window,text=" ",font=("Courier",30)).grid(row=5,column=1)
tk.Button(window,text="Restart Game",command=restart_program).grid(row=6,column=1)





def gamerules(bot_choice,player_choice):
	if(bot_choice==player_choice):
		return "Draw"
	elif(bot_choice=="paper" and player_choice == "rock"):
		return "Bot Wins"
	elif(bot_choice=="rock" and player_choice == "paper"):
		return "Player Wins"

	elif(bot_choice == "paper" and player_choice == "scissors"):
		return "Player Wins"
	elif(bot_choice == "scissors" and player_choice == "paper"):
		return "Bot Wins"

	elif(bot_choice == "rock" and player_choice == "scissors"):
		return "Bot Wins"
	elif(bot_choice == "scissors" and player_choice == "rock"):
		return "Player Wins"

def countdown(counttime=10):
	global waktu,player_choice,bot_choice
	countdowntk_label=tk.Label(window,text="")
	if(waktu is None):
		waktu=counttime	
	if waktu<=0:
		countdowntk_label=tk.Label(window,text="Times up",font=("Courier",30))
		countdowntk_label.grid(row=3,column=1)
		tk.Label(window,text=str(bot_choice)+" v.s "+str(player_choice)).grid(row=4,column=1)
		gameresult=gamerules(bot_choice,player_choice)
		tk.Label(window,text=gameresult,font=("Courier",30)).grid(row=5,column=1)
	else:
		waktu=waktu-1
		countdowntk_label.after(1000,countdown)
		countdowntk_label=tk.Label(window,text=str(waktu),font=("Courier",30))
		countdowntk_label.grid(row=3,column=1)
	tk.Label(window,text="Bot Hands").grid(row=0,column=0)

def bot_func():
	global waktu,bot_choice
	bot_choice=bot_hands[random.randint(0,2)]
	bot_hands_image=cv2.imread(bot_img[bot_choice])
	tk_imagebot=cv2.cvtColor(bot_hands_image, cv2.COLOR_BGR2RGB)
	tk_imagebot=Image.fromarray(tk_imagebot)
	botimgtk=ImageTk.PhotoImage(image=tk_imagebot)
	botimgtk_label=tk.Label(image=botimgtk,anchor="n")
	botimgtk_label.image=botimgtk
	botimgtk_label.grid(row=1,column=0)
	
	if(bot_choice!="scissors"):
		tk.Label(window,text="  "+bot_choice+"  ",font=("Courier",30)).grid(row=3,column=0)
	else:
		tk.Label(window,text=bot_choice,font=("Courier",30)).grid(row=3,column=0)
	if waktu>0:
		botimgtk_label.after(100,bot_func)
	else:
		pass
def show_frame():
	global waktu,player_choice,player_image
	with mp_hands.Hands(
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
			max_num_hands=1) as hands:
		success, image = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
		h,w,c=image.shape
		image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
		rawimage=image[:]
		rawimage = cv2.cvtColor(rawimage, cv2.COLOR_RGB2BGR)
		image.flags.writeable = False
		results = hands.process(image)
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		image2=np.zeros((300,300,3),np.uint8)	
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
				x_mid,y_mid=int(x_min+((x_max-x_min)/2)),int(y_min+((y_max-y_min)/2))
				panjang,lebar=x_max-x_min,y_max-y_min
				panjang=panjang+80
				lebar=lebar+80
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
				image2 = cv2.resize(image2, (300,300), interpolation = cv2.INTER_AREA)
				koordinat=[]
				for num,i in enumerate(hand_landmarks.landmark):
					if(panjang>lebar):
						x_landmark=((i.x*w)-(x_mid-int(panjang/2)))/panjang
						y_landmark=((i.y*h)-(y_mid-int(panjang/2)))/panjang
						txt=str(num)+" x="+str(int(i.x*w)-(x_mid-int(panjang/2)))+",y="+str(int(i.y*h)-(y_mid-int(panjang/2)))
					if(lebar>panjang):
						x_landmark=((i.x*w)-(x_mid-int(lebar/2)))/lebar
						y_landmark=((i.y*h)-(y_mid-int(lebar/2)))/lebar
						txt=str(num)+" x="+str(int(i.x*w)-(x_mid-int(lebar/2)))+",y="+str(int(i.y*h)-(y_mid-int(lebar/2)))
					try:
						cv2.putText(image,txt,(int(i.x*w),int(i.y*h)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
						koordinat.append(int(x_landmark*300))
						koordinat.append(int(y_landmark*300))
					except:
						pass
				if waktu>0:
					try:
						predicted=model.predict([koordinat])
						player_choice=predicted[0]
						if(predicted[0]!="scissors"):
							tk.Label(window,text="  "+player_choice+"  ",font=("Courier",30)).grid(row=3,column=2)
						else:
							tk.Label(window,text=player_choice,font=("Courier",30)).grid(row=3,column=2)
						player_image=image2
					except:
						pass
				else:
					pass
		tk.Label(window,text="Camera Feed").grid(row=0,column=1)
		timage=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		timage=Image.fromarray(timage)
		imgtk=ImageTk.PhotoImage(image=timage)
		imgtk_label=tk.Label(image=imgtk)
		imgtk_label.image=imgtk
		imgtk_label.grid(row=1,column=1)
		tk.Label(window,text="Predicted Gesture").grid(row=2,column=2)
		tk.Label(window,text="Player Hand").grid(row=0,column=2)
		timage2=cv2.cvtColor(player_image, cv2.COLOR_BGR2RGB)
		timage2=Image.fromarray(timage2)
		imgtk2=ImageTk.PhotoImage(image=timage2)
		imgtk_label2=tk.Label(image=imgtk2,anchor="n")
		imgtk_label2.image=imgtk2
		imgtk_label2.grid(row=1,column=2)
		

		imgtk_label.after(10,show_frame)



countdown(10)
bot_func()
show_frame()
window.mainloop()