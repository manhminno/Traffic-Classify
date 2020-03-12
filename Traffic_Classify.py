import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


def read_data(label2id):
    X = []
    Y = []
    for label in os.listdir('Traffic-Data/trainingset'):
        for img_file in os.listdir(os.path.join('Traffic-Data/trainingset', label)):
            img = cv2.imread(os.path.join('Traffic-Data/trainingset', label, img_file))            
            X.append(img)
            Y.append(label2id[label])
    return X, Y


# Label to id, used to convert string label to integer 
label2id = {'pedestrian':0, 'moto':1, 'truck':2, 'car':3, 'bus':4}
X, Y = read_data(label2id)


#Trich xuat dac trung
def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.xfeatures2d.SIFT_create()

    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)

    return image_descriptors

image_descriptors = extract_sift_features(X)


#Xay dung tu dien
all_descriptors = []
for descriptors in image_descriptors:
    if descriptors is not None:
        for des in descriptors:
            all_descriptors.append(des)

def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_
    return bow_dict

num_clusters = 100

if not os.path.isfile('Traffic-Data/bow_dictionary150.pkl'):
    BoW = kmeans_bow(all_descriptors, num_clusters)
    pickle.dump(BoW, open('Traffic-Data/bow_dictionary150.pkl', 'wb'))
else:
    BoW = pickle.load(open('Traffic-Data/bow_dictionary150.pkl', 'rb'))


#Xay dung vecto dac trung tu dict
def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []
    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance, axis=1)
            for j in argmin:
                features[j] += 1
        X_features.append(features)
    return X_features

X_features = create_features_bow(image_descriptors, BoW, num_clusters)


#Xay dung model
X_train = [] 
X_test = []
Y_train = []
Y_test = []
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.2, random_state=42)

svm = sklearn.svm.SVC(C = 10)
svm.fit(X_train, Y_train)

#Thu predict 
img_test = cv2.imread('Traffic-Data/image_test/car.png')
img = [img_test]
img_sift_feature = extract_sift_features(img)
img_bow_feature = create_features_bow(img_sift_feature, BoW, num_clusters)
img_predict = svm.predict(img_bow_feature)

print(img_predict)
for key, value in label2id.items():
    if value == img_predict[0]:
        print('Your prediction: ', key)

#Accuracy
print(svm.score(X_test, Y_test))

#Show image
cv2.imshow("Img", img_test)
cv2.waitKey()
