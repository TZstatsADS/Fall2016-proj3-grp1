
# coding: utf-8

# In[53]:

import cv2
import numpy as np
import matplotlib as plt
from glob import glob
get_ipython().magic(u'matplotlib inline')
from sklearn.cluster import KMeans


# In[97]:

#return a list of image filenames
def get_img_name(file_path = './data/images/', file_type = 'jpg'):
    file_path, file_type = '../data/images/','jpg'
    img_names = glob('{0}*.{1}'.format(file_path, file_type))
    return img_names

#testing the availability of image at the same time
def label_img(img_name_list):
    count = 0
    img_set, labels = [], []
    for i in img_name_list:
        #Test whether the image is corrupted
        #This part can be skipped to speed up the process
        test_read = cv2.imread(i)
        if test_read is None:
            count += 1
            continue
        img_set.append(i)
        #Chicken => 0 Dog => 1
        labels.append(0 if i.split('/')[-1][0].lower()=='c' else 1)
    print '{} image(s) is(are) corrupted!'.format(count)
    return np.array(img_set), np.array(labels)    


# In[98]:

#SIFT Feature extraction
def sift_extract(image_list, n = 50, conThre = 0.04, edgeThre = 10):
    raw_features = []
    for i in image_list:   
        img = cv2.imread(i,0) #read in as grey scale
        #bunch of parameter to be tuned
        sift = cv2.xfeatures2d.SIFT_create(nfeatures = n,
                                          contrastThreshold = conThre,
                                          edgeThreshold = edgeThre)
        kps, descs = sift.detectAndCompute(img, None)
        raw_features.append(descs)
    return raw_features
    
#-------for test purpose, select part of the data
#import random 
#index  = random.sample(range(len(images)),50)
#mages = images[index]
#labels = labels[index]
#test_feature_list = sift_extract(images)


# In[99]:

#combine features into a ndarray and return a vector of group label   
def combine_feature(raw_feature_list):
    feature_array = np.vstack(raw_feature_list)
    group_number = [x.shape[0] for x in raw_feature_list]
    group_label = np.repeat(range(len(raw_feature_list)), group_number)
    return feature_array, group_label
    
    
def kmeans_cluster(raw_feature_list, n_cluster = 1000, max_Iter = 50):
    features = np.zeros((len(raw_feature_list), n_cluster))
    feature_array, group_label = combine_feature(raw_feature_list)
    #dtype check, same RAM
    if not feature_array.dtype == 'float32':
        feature_array.dtype == 'float32'
    
    km = KMeans(n_clusters = n_cluster, max_iter = max_Iter).fit(feature_array)
    cluster_label = km.labels_
    #assign every feature (row) to its corresponding group
    for i in range(feature_array.shape[0]):
        features[group_label[i],cluster_label[i]] += 1
    #convert to probability
    features = np.apply_along_axis(lambda x: x/sum(x), 1, features)
    return features

    


# In[ ]:

#Main 
def main():
    images, labels = label_img(get_img_name())
    raw_features = sift_extract(images)
    return kmeans_cluster(raw_features)
get_ipython().magic(u'time result = main()')

