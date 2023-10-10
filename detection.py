import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

sys.path.append("Mask_RCNN-Multi-Class-Detection")#add the maskrcnn to the system path
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn import *
from mrcnn import visualize
from mrcnn import utils
import warnings
from xml.etree import ElementTree
import skimage.draw
import cv2
import imgaug
import numpy as np
import math
# import keras libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import queue
from fpdf import FPDF

class PredictionConfig(Config):# creat a custom class for the maskrcnn model inserting the new configuration
    NAME = "coco"
    NUM_CLASSES = 81
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


global prev#define a global variable named prev that will save the last saved time of the current status
prev = None# set initial value to be none
print(tf.test.is_gpu_available())#check if the code is utilizing GPU
warnings.filterwarnings("ignore")
video_dir = "parkinglot_1_480p.mp4"# the video directory
model_path = 'Mask_RCNN-Multi-Class-Detection/mask_rcnn_coco.h5'# the pre trained model weights file

my_queue = queue.Queue()#define a queue that will save results from the multithreaded function that saves PDF fils
cap = cv2.VideoCapture(video_dir)#read video feed

ret,frame=cap.read()# get the first frame from the video feed ret is a boolen that shows if there is any feed from the feed or not
h,w,c=frame.shape#geet height and width and number of channels in the frame
result = cv2.VideoWriter('motion.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         30, (w,h))#set the video saver to save videos with the same dimensions of the frmae and in mp4 format
cfg = PredictionConfig()#create an objects of the PredictionConfig class that we created

model = MaskRCNN(mode='inference', model_dir='./', config=cfg)#create a Model in infrence mode(not a training code) with the previous configurations

model.load_weights(model_path, by_name=True)#load the weights of the model from the directory
with open("coco_names.txt", 'r') as f:#open a txt file with the names of the classes of the coco dataset
    class_names = f.readlines()#read the lines and save in a list
class_names.insert(0, 'background')#first class is background

try:#check if there is a file for predefined parking spaces
    with open('parking_lots.pkl', 'rb') as f:
        done_polys = pickle.load(f)
except:
    done_polys = []

parking_masks = []# a list that will save all the masks of the parking spots
availble = []#availability list


def storeInQueue(f):#a function that stores variables in the queue
  def wrapper(*args):
    my_queue.put(f(*args))
  return wrapper

@storeInQueue
def save_state(availbality,prev):#this function will run on a thread i wnat it to have acceses to the store in queue function
    if prev is not None:# if there was a previous time that the status was saved
        delta = datetime.now() - prev#get diffrence in time
        if (delta.total_seconds() / 60) >= 2:#if the delta is greater that 2 mins
            availble_spaces = np.count_nonzero(np.array(availbality))# count nonzero elemnts in the availability list
            taken_spaces = len(availbality)-availble_spaces#get taken spaces
            text = f"Time: {datetime.now()}\nAvailble spaces: {availble_spaces}\nTaken spaces:  {taken_spaces}\nTotal number of spaces : {len(availbality)}"#message to be saved in the pdf
            pdf = FPDF()# create an object o fthe FPDF class
            pdf.add_page()#add page to the pdf
            pdf.set_font("Arial", size=15)#define font properties
            pdf.multi_cell(0, 5, txt=text,align='C')#create a multiline cell with centered alligmnet
            now = datetime.now()  # current date and time#get current time
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")#preprocessing for text to have good formate
            date_time = ":".join(date_time.split("/")).split(":")
            date_time = "_".join(date_time)
            date_time = "_".join(date_time.split(","))
            pdf.output(f"Garage_status_{date_time}.pdf")# save file as Garage_status_ follwed by current date and current time

            prev = datetime.now()#redefine previous to current time since for next time this will be the previous

    else:
        prev = datetime.now()#if previous is None  make as current time
    return prev#return prev


def binaryMaskIOU(mask1, mask2):#a function that calculates the intersection over union between cars and parking spots

    mask1_area = np.count_nonzero(mask1 == 1)  # get area of first mask
    mask2_area = np.count_nonzero(mask2 == 1)

    intersection = np.count_nonzero(np.bitwise_and(mask1, mask2)) # get intersection

    iou = intersection / (mask1_area + mask2_area - intersection) # get intersection over union
    return iou # return intersection over union


def detect_cars(img,parking_masks):
    h, w, c = img.shape# get image shape
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# transofrm image colors from the bGR formate of opencv to RGB formate
    image = np.array(image).reshape(-1, h, w, c)#transform image to a numpy array and reshape it so it is 1,h,w,c as maskrcnn accepts the images in batches so we pass it as a batch of one image
    results = model.detect(image, verbose=0)#detect cars from the image
    r = results[0]# get first result as we said maskrcnn accepts batch of images it also returns a batch of results since we have one iimage there is a batch of one result which is the one we are intersted in
    object_count = len(r["class_ids"])# get number of detected cars
    tot_mask = np.zeros_like(img)#get total mask of all cars and slots

    for ind, parking_mask in enumerate(parking_masks):# for each parking slot
        flag = False
        cnts = cv2.findContours(parking_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]#get parking slot contours
        for i in range(object_count):#check if there is a car found inside
            mask = r["masks"][:, :, i]#get car maask
            mask = np.array(mask).astype(np.uint8)#transform mask int right foramte to compare to parking spot mask
            iou = binaryMaskIOU(mask,parking_mask)#get the IOU
            if iou>=0.15:# if iou is greater tha a threshhold of 0.15
                mask_zero = np.zeros_like(img)#create a zero mask same shape as the image
                cv2.fillPoly(mask_zero, [cnts[0]], (0, 0, 255))#fill the empty mask with parking slot contours in red since there is a car inside
                tot_mask = cv2.add(tot_mask,mask_zero)# add mask to the toal mask
                availble[ind]=0#set availbility to 0
                flag = True
                break# break out of the loop since this spot has been found unavailble already
        if not flag:# if there was no car found
            availble[ind] = 1#set availbility to one
            mask_zero = np.zeros_like(img)

            cv2.fillPoly(mask_zero, [cnts[0]], (0, 255, 0))#fill empty mask with green
            tot_mask = cv2.add(tot_mask, mask_zero)

    #cv2.imshow("mask",tot_mask)
    img = cv2.addWeighted(img,0.7,tot_mask,0.4,0.0)#add tot mask to the image with a 0.4 weight
    cv2.rectangle(img,(0,0),(270,60),(0,0,0),-1)#show a black rectangle top left of the screen
    availble_spaces = np.count_nonzero(np.array(availble))#count number of availble spaces
    text = f" Availble: {availble_spaces}/{len(availble)}"# create text message
    cv2.putText(img,text,(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)# show availble and total spaces on the screen
    return img,availble#return resulted image and avaialbility list

for poly in done_polys:#fill parking masks according to the points feeded from the pickel file
    points = np.int32(poly).reshape(-1, 2)

    mask = np.zeros_like(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    cv2.fillPoly(mask,[points],1)
    parking_masks.append(mask)
    availble.append(0)

while True:#infite loop
    ret, frame = cap.read()#get frames from the video feed
    if not ret:# if the video feed ended
        break
        #cap = cv2.VideoCapture(video_dir)#reload the video
        #ret, frame = cap.read()
    frame,availble = detect_cars(frame,parking_masks)#run the detect cars function that was explained above
    my_thread = Thread(target=save_state, args=(availble,prev))#run the save state function on another thread
    my_thread.start()
    prev = my_queue.get()#get prev
    cv2.imshow("frame", frame)#show frame
    result.write(frame)#save video frame to video output
    key = cv2.waitKey(1)#set a delay of millisecond that reads keyboard feed

    if key & 0xFF == ord("q"):# if "q" is pressed break out of the infinite loop
        break
cap.release()
cv2.destroyAllWindows()