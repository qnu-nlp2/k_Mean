# Cài đặt thuật toán k-Mean để phân k (3<=k<=10) cụm khách hàng  từ dữ liệu Shoping_CustomerData.csv
# K-Means-Clustering
Unsupervised learning - exclusive clustering

This is an algorithm to create exclusive clusters. 

**Mathematical Intuition and Notes**

Mathematically expressed as:
* Let C1, C2, Ck be the K clusters
* Then we can write: $C_1 U C_2 UC_3 U…UC_k $={1,2,3,…,n} i.e., each datapoint has been assigned to a cluster.
* The idea behind the K-Means clustering approach is that the within-cluster variation amongst the point should be minimum. The within-cluster variance is denoted by: W(Ck).Hence, according to the statement above, we need to  minimize this variance for all the clusters. 
* The next step  is to define the criterion for measuring the within-cluster variance. Generally, the criterion is the Euclidean distance between two data points.
* The above formula says that we are calculating the distances between all the point in a cluster, then we are repeating it for all the K clusters(That’s why two summation signs) and then we are dividing it by the number of observation in the clusters (Ck is the number of observations in the Kth cluster) to calculate the average.

So, ultimately our goal is to minimize the distance of 'tagged' data points from the centroids. 

The following algorithm steps are used to solve this problem.

**Algorithm**

1. Randomly assign K cluster centers. We make sure that these are very distant from each other. 
2. Calculate the distance of all the data points from all the K number of centers and allocate the points to the cluster based on the shortest distance. The model's *inertia* is the mean squared distance between each instance and its closest Kth point. The K points are also called **centroids** which we have randomly selected in step 1. Our goal is the have a model with lowest inertia. 
3. Recompute the centroids (location) once all points are assigned to the nearby Kth centroid. 
4. Repeat steps 2 and 3, until the locations of the centroid strop changing and the cluster allocation of the points become constant! 

**Upfront problems**

1. How to choose right value for K? **ANS = ELBOW METHOD** 
2. How to make it performance friendly? **ANS = K-Means++**

###### The Elbow-Method

This method is based on the relationship between the within-cluster sum of squared distances(WCSS Or Inertia) and the number of clusters. It is observed that first with an increase in the number of clusters WCSS decreases steeply and then after a certain number of clusters the drop in WCSS is not that prominent. The point after which the graph between WCSS and the number of clusters becomes comparatively smother is termed as the elbow and the number of cluster at that point are the optimum number of clusters as even after increasing the clusters after that point the variation is not decreasing by much i.e., we have accounted for almost all the dissimilarity in the data.

###### Challenges and improvements in K-Means:

1.	We need to specify the number of clusters beforehand.
2.	It is required to run the algorithm multiple times to avoid a sub-optimal solution
3.	K-Means does not behave very well when the clusters have varying sizes, different densities, or non-spherical shapes.
4.	The clusters sometimes vary based on the initial choice of the centroids. An important improvement to the K-Means algorithm, called **K-Means++**, was proposed in a _2006 paper by David Arthur and Sergei Vassilvitskii_. They introduced a smarter initialization step that tends to select centroids that are distant from one another, and this makes the K-Means algorithm much less likely to converge to a suboptimal solution.
5.	Another important improvement to the K-Means algorithm was proposed in a _2003 paper by Charles Elkan_.It considerably accelerates the algorithm by avoiding many unnecessary distance calculations: this is achieved by exploiting the _triangle inequality_ (i.e., the straight line is always the shortest; in a triangle with sides a,b and c=> **a+b>c**) and by keeping track of lower and upper bounds for distances between instances and centroids. 
6.	Yet another important variant of the K-Means algorithm was proposed in a _2010 paper by David Sculley_. Instead of using the full dataset at each iteration, the algorithm is capable of using **mini-batches**, moving the centroids just slightly at each iteration. This speeds up the algorithm typically by a factor of 3 or 4 and makes it possible to cluster huge datasets that do not fit in memory. Scikit-Learn implements this algorithm in the **MiniBatchKMeans** class. You can just use this class like the KMeans class

#**Demo: K-Means Clustering Using Python**

###**Problem Definition**

Perform K-Means Clustering on a dataset containing shopping details of customers from various cities to understand how clustering segregates the data similar to each other.




###**Dataset Description**

The dataset is based on the the shopping details of customers of few cities. The data has been populated randomly and holds no such relation to any real systems data or confidential data. It has 8 dimensions or features with 200 entries.

The Attributes are:

>* CustomerID
>* CustomerGender
>* CustomerAge
>* CustomerCity
>* AnnualIncome
>* CreditScore
>* SpendingScore
>* CustomerCityID

###**Tasks to be performed**


>* Importing Required Libraries
>* Analyzing the data
>* Understanding K-Means
>* Implementing K-Means from Scratch
>* Implementing K-Means using sklearn library 

###**Importing Required Libraries**



```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,confusion_matrix
import warnings
warnings.filterwarnings("ignore")
```
