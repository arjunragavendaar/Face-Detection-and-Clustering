'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import cv2
import numpy as np
import os
import sys
import math

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = [] # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.

    img_grey_image=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    final_width,final_height=0,0

    # face_detect_haar = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    # face_detect_haar_new=cv2.CascadeClassifier()
    # face_detect_haar_new.load(face_detect_haar)
    
    # detect_more_faces= face_detect_haar_new.detectMultiScale(img_grey_image,1.09,10,minSize=(34,34),flags=cv2.CASCADE_SCALE_IMAGE)
    detect_more_faces=face_recognition.face_locations(img_grey_image,number_of_times_to_upsample=1)
    for (top,right,bottom,left) in detect_more_faces:
        
        # obtaining the width and height from the four points 
        final_width=right-left
        final_height=bottom-top


        if right >= img_grey_image.shape[1]:
            new_val=(right-img_grey_image.shape[1])+1
            final_width=right-left-new_val

        if bottom >= img_grey_image.shape[0]:

            new_val=(bottom-img_grey_image.shape[0])+1
            final_height=bottom-top-new_val
        #final width and height
        final_right=left+final_width
        final_bottom=top+final_height
        # print(left,top,final_right,final_bottom)
        cv2.rectangle(img_grey_image,(left,top),(final_right,final_bottom),(29,29,29),2) 
        detection_results.append([float(left),float(top),float(final_width),float(final_height)])
        # show_image(img_grey_image)
    return detection_results


def cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    cluster_results: List[List[str]] = [[]] * K # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.
    resultant_vector=[]
    evaluation_data={}
    for key in imgs:
        box_values=detect_faces(imgs[key])
        box_val=box_values[len(box_values)-1]
        top=box_val[1]
        left=box_val[0]
        bottom=box_val[1]+box_val[3]
        right=box_val[0]+box_val[2]
        intermediate_result=face_recognition.face_encodings(imgs[key],[(int(top),int(right),int(bottom),int(left))])
        resultant_vector.append(intermediate_result[0].reshape(1,128))
        evaluation_data[key]=intermediate_result[0].reshape((1,128))
    #kmeans initialize
    kmeans_initialize=KMeans(K)
    reshaped_result=np.array(resultant_vector).reshape((len(imgs),128))
    kmeans_initialize.train_image_data(reshaped_result)
    kmeans_eval=kmeans_initialize.eval_data(evaluation_data)
    for itr in kmeans_eval:
        cluster_results[itr]=kmeans_eval[itr]

    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''
def get_distance(data_1,data_2):
    result=np.sqrt(np.sum((data_1-data_2)**2,axis=1))
    return result   

class KMeans():
    def __init__(self,k,max_iter=300):
        np.random.seed(102)
        self.maximum_iteration=max_iter
        self.numberOfclusters=k
        # self.finaloutput=[]
        # for i in range(self.numberOfclusters):
        #     self.finaloutput[i]=[]

        self.finaloutput={i:[] for i in range(self.numberOfclusters)}

        self.centroids=[]

    def train_image_data(self,resultant_vector):

        self.centroids=resultant_vector[np.random.choice(len(resultant_vector),self.numberOfclusters,replace=False),:]
        previous_centroid_value=[]
        intermediate_value_holder=[]
        itr=0

        for row in range(self.numberOfclusters):
            intermediate_value_holder=[]
            for column in range(resultant_vector.shape[1]):
                intermediate_value_holder.append(0)
            previous_centroid_value.append(intermediate_value_holder)

            image_cluster=[]
        while itr<self.maximum_iteration and np.not_equal(self.centroids,previous_centroid_value).any():
     
            for i in range(self.numberOfclusters):
                val=[]
                image_cluster.append(val)

            for data in resultant_vector:
                img_dist=get_distance(data,self.centroids)
                centriod_val=np.argmin(img_dist)
                image_cluster[centriod_val].append(data)

            for cluster in range(self.numberOfclusters):
                image_cluster[cluster]=np.array(image_cluster[cluster])  

            previous_centroid_value=self.centroids   

            for x in range(len(image_cluster)):
                self.centroids[x]=np.mean(image_cluster[x],axis=0)

            for current_centroid in range(len(self.centroids)):
                if np.isnan(self.centroids[current_centroid]).any():
                    self.centroids[current_centroid]=previous_centroid_value[current_centroid]
            
            self.centroids=np.array(self.centroids)
            itr=itr+1
    def predict_data(self,param):
        distval=get_distance(param,self.centroids)
        minimum_val=np.argmin(distval)
        return minimum_val

    def eval_data(self,param):
        for key in param:
            predict_val=self.predict_data(param[key])
            self.finaloutput[predict_val].append(key)

        return self.finaloutput

# Your functions. (if needed)
