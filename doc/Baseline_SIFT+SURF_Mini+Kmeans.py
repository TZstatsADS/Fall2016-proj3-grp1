
# coding: utf-8

# In[1]:

import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
get_ipython().magic(u'matplotlib inline')
import random
import time


# In[92]:

#return a list of image filenames
def get_img_name(f_path = '../data/images/', f_type = 'jpg'):
    file_path, file_type = f_path, f_type
    img_names = glob('{0}*.{1}'.format(file_path, file_type))
    return img_names

#testing the availability of image at the same time
def label_img(img_name_list):
    count = 0
    img_set, labels = [], []
    for i in img_name_list:
        #Test whether the image is corrupted
        #This part can be skipped to speed up the process
        #test_read = cv2.imread(i)
        #if test_read is None:
        #    count += 1
        #    continue
        img_set.append(i)
        #Chicken => 0 Dog => 1
        labels.append(0 if i.split('/')[-1][0].lower()=='c' else 1)
    #print '{} image(s) is(are) corrupted!'.format(count)
    return np.array(img_set), np.array(labels)    

#img_names = get_img_name()
#image_list, labels = label_img(img_names)


# In[64]:

#SIFT Feature extraction
#------Note on Feature extraction ---------
#input format: list of image names ; output format: list of 2000 ndarrays with 128 columns(in both SIFT and SURF)
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

#%time raw_features = sift_extract(image_list)


# In[121]:

#SUFT Feature extraction
def suft_extract(image_list, hessian = 4000):
    raw_features = []
    for i in image_list:
        img = cv2.imread(i, 0)
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold = hessian, extended = True)
        kp, descs = surf.detectAndCompute(img, None)
        
        #for specific image if number of keypoints is 0 (hessian value too big),
        #gradually reduce hessian value by 500 untill there're keypoints detected from the image
        temp_hessian = hessian
        while descs is None:
            temp_hessian -= 500
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold = temp_hessian, extended = True)
            kp,descs = surf.detectAndCompute(img, None)
    
        
        raw_features.append(descs)
    return raw_features

#%time raw_features = suft_extract(image_list, hessian = 5000)


# In[38]:

#combine features into a single ndarray and return a vector of group label   
#group label used for mapping the result cluster center into corresponding image
def combine_feature(raw_feature_list):
    feature_array = np.vstack(raw_feature_list)
    group_number = [x.shape[0] for x in raw_feature_list]
    group_label = np.repeat(range(len(raw_feature_list)), group_number)
    return feature_array, group_label

#---------Note on clustering---------
#input format: list of ndarray, output format single ndarray with dimension(n_image, n_cluster)
def kmeans_cluster(raw_feature_list, sub_sample = False, sub_perc = 0.5, n_cluster = 1000, max_Iter = 50):
    from sklearn.cluster import KMeans
    
    features = np.zeros((len(raw_feature_list), n_cluster))
    feature_array, group_label = combine_feature(raw_feature_list)
    #dtype check, same RAM
    if not feature_array.dtype == 'float32':
        feature_array.dtype == 'float32'
    
    #if number of features is larger than sub_perc*feature number and sub_sample is on
    #randomly subsample features
    if sub_sample and (sub_perc * feature_array.shape[0])>n_cluster:
        n_row  = feature_array.shape[0]
        sub_sample_index = random.sample(range(n_row), int(sub_perc*n_row))
        feature_array = feature_array[sub_sample_index,:]
        group_label = group_label[sub_sample_index]

    #In the case when cluster number is larger than data dimention
    #change cluster number to half of the feature 
    if feature_array.shape[0] < n_cluster:
        n_cluster = feature_array.shape[0]/2
        
    km = KMeans(n_clusters = n_cluster, max_iter = max_Iter).fit(feature_array)
    cluster_label = km.labels_
    #assign every feature (row) to its corresponding group
    for i in range(feature_array.shape[0]):
        features[group_label[i],cluster_label[i]] += 1
    #convert to probability
    features = np.apply_along_axis(lambda x: x/sum(x), 1, features)
    return features

#%time result = kmeans_cluster(raw_features, sub_sample=True, n_cluster=100)


# In[49]:

def mini_kmeans(raw_feature_list, n_cluster = 1000, sample_size = 100, Iter = 50):
    #Randomly sample between for each iteration
    from sklearn.cluster import MiniBatchKMeans
    
    features = np.zeros((len(raw_feature_list), n_cluster))
    feature_array, group_label = combine_feature(raw_feature_list)
    #dtype check, same RAM
    if not feature_array.dtype == 'float32':
        feature_array.dtype == 'float32'
        
    #In the case when cluster number is larger than data dimention
    #change cluster number to half of the feature 
    if feature_array.shape[0] < n_cluster:
        n_cluster = feature_array.shape[0]/2
        
    mini_kmeans = MiniBatchKMeans(n_clusters = n_cluster, batch_size = sample_size, 
                                  max_iter = Iter).fit(feature_array)
    cluster_label = mini_kmeans.labels_
    #assign feature to group
    for i in range(feature_array.shape[0]):
        features[group_label[i],cluster_label[i]] += 1
    #convert to probability
    features = np.apply_along_axis(lambda x: x/sum(x), 1, features)
    return features

#%time result = mini_kmeans(raw_features, n_cluster = 100)


# In[123]:

#return train data
def preprocess(f_path = '../data/images/', f_type = 'jpg',
               feature_method = 'SIFT', n_keypoint = 50, hessian_thre = 3000,
               cluster_method = 'KMeans',sub = False, perc = 0.5, clusters = 1000, Iter = 50, sample_size = 100):
    images, labels = label_img(get_img_name(f_path = f_path, f_type = f_type))
    if feature_method == 'SIFT':
        raw_features = sift_extract(images, n = n_keypoint)
    elif feature_method == 'SURF':
        raw_features = suft_extract(images, hessian= hessian_thre)
    if cluster_method == 'KMeans':
        return kmeans_cluster(raw_features,sub_sample = sub , sub_perc = perc, n_cluster = clusters, max_Iter = Iter), labels
    elif cluster_method == 'MiniKMeans':
        return mini_kmeans(raw_features, n_cluster=clusters, sample_size = sample_size, Iter = Iter), labels
    


# In[125]:

#TEST USING SVM

get_ipython().magic(u"time x, y = preprocess(clusters = 6000, feature_method='SURF',cluster_method = 'MiniKMeans')")
import pandas as pd
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.svm import SVC
#base_sift = pd.read_csv('/Users/Max/GitHub/Fall2016-proj3-grp1/data/sift_features.csv')
#base_sift = base_sift.transpose()
#x = base_sift.values
    
rs = ShuffleSplit(n_splits = 3, test_size = 0.3, random_state=0)

score = []
for train_index, test_index in rs.split(x):
    train_x, train_y = x[train_index,:], y[train_index]
    test_x, test_y = x[test_index,:], y[test_index]
    clf = SVC(gamma = 1, C = 100)
    clf.fit(train_x, train_y)
    score.append(clf.score(test_x, test_y))
print np.mean(score)


# In[ ]:

#run preprocess, feed data to R
#if __name__ == '__main__':
#    features, labels = preprocess(f_path= '/Users/Max/Downloads/test/')
    


# In[ ]:



