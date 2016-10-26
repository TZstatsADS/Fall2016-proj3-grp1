
# coding: utf-8

# In[48]:

from numpy import genfromtxt
import numpy as np
my_data = genfromtxt('/Users/mac/Downloads/ADS/sift_features.csv', delimiter=',')
#my_data1 =  /Users/mac/Downloads/sift_image.pynp.asmatrix(my_data)


# In[50]:

my_data = np.array(my_data)
my_data1 = np.delete(my_data, (0), axis=0)


# In[70]:

label = np.zeros(2000)
label[1000:2000] = 1
my_data2 = np.vstack((label,my_data1))


# In[94]:

my_data3 = np.transpose(my_data2)


# In[103]:

my_data4 = my_data3[np.random.choice(len(my_data3), len(my_data3), replace=False),:]


# In[116]:

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.qda import QDA

x = my_data4[:,1:]
y = np.asarray(my_data4[:,:1])

X_train, X_test, y_train, y_test = train_test_split(
     x, y, test_size=0.9, random_state=0)


clf1 = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print clf1.score(X_test, y_test) 

clf2 = QDA()
clf2.fit(X_train, y_train)
print clf2.score(X_test, y_test)



# In[ ]:




# In[119]:

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf3 = RandomForestClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
scores = cross_val_score(clf3, X_train, np.asarray(y_train))
scores.mean() 


# In[122]:




# In[ ]:



