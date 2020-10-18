import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import sys
import time

#Sample size points from the uniform distribution over a d-dimensional ball

#dim: dimension of the desired sample
#size: size of the sample
def uniform_ball(dim,size):
	mean = [0]*dim
	cov = np.identity(dim)
	angle = np.random.multivariate_normal(mean,cov,size=size)
	rad = np.random.uniform(size=size)
	norm = np.linalg.norm(angle,axis=1)
	scaling_factor = pow(rad,1.0/dim)/norm

	return np.hstack((angle*np.transpose(np.array([scaling_factor])),np.transpose(np.array([[1]*size]))))

#Draw a random d-dimensional non-homogenous hyperplane
#and label the inputted points.

#points: points to be labeled by the classifier
#dim: dimension of the classifier
#worst: determines whether shift is random, or set to be on the sphere.
def classifier(points,dim,worst=False):
	mean = [0]*dim
	cov = np.identity(dim)
	angle = np.random.multivariate_normal(mean,cov,size=1)
	angle = angle/np.linalg.norm(angle)
	if worst:
		b=[1]
	else:
		b = np.random.uniform(low=-1,high=1,size=1)
	angle = np.hstack((angle,[b]))
	labels = np.dot(points,np.transpose(angle))
	return angle,np.hstack((points,np.sign(labels))),labels

#Build and run the LP for inference on points[uninferred].

#points: overall sample
#uninferred: indices of points which have not been inferred
#constraints: a list of indices of subsamples 
#angle: the classifier
#query: either "label" or "comp," the type of query
def infer(points,uninferred,constraints,angle,query):
	A = []
	b2 = []
	inferred = []
	pos = []
	neg = []
	for j in range(len(constraints)):

		for i in range(len(constraints[j])):

			label = np.sign(np.dot(angle,np.transpose(np.array([points[constraints[j][i]]]))))
			if label == -1:
				A.extend([points[constraints[j][i]]])
			else:
				A.extend(-points[[constraints[j][i]]])
			b2.extend([-0.0000001])
			# If we are using comparisons, add in comparison constraints
			if query == "comp":
				for k in range(i+1,len(constraints[j])):
					label_comp = np.sign(np.dot(angle,np.transpose(np.array([points[constraints[j][i]]-points[constraints[j][k]]]))))
					if label_comp == -1:
						A.extend([points[constraints[j][i]]-points[constraints[j][k]]])
					else:
						A.extend([points[constraints[j][k]]-points[constraints[j][i]]])
					b2.extend([-0.0000001])
	for i in uninferred:
		c = points[i]
		val_p = scipy.optimize.linprog(c,A_ub=np.array(A),b_ub=np.array(b2),bounds=(-2,2))
		val_n = scipy.optimize.linprog(-c,A_ub=np.array(A),b_ub=np.array(b2),bounds=(-2,2))
		if val_p.fun > 0:
			inferred.extend([i])
			pos.extend([i])
		elif val_n.fun > 0:
			inferred.extend([i])
			neg.extend([i])
		else:
			continue
	return inferred, pos, neg

# This Mergesort comparison counter is adapted from:
# https://stackoverflow.com/questions/42608630/how-to-add-a-comparison-counter-for-merge-sort-in-python
def MergeSort(argShuffledList):
    intNumOfComp = 0

    if len(argShuffledList)>1:
        intMidValue = len(argShuffledList)//2
        listLeftHalf = argShuffledList[:intMidValue]
        listRightHalf = argShuffledList[intMidValue:]

        left_part = MergeSort(listLeftHalf)
        right_part = MergeSort(listRightHalf)

        intNumOfComp += left_part[1] + right_part[1]

        i=0
        j=0
        k=0
        while i < len(listLeftHalf) and j < len(listRightHalf):
            intNumOfComp += 1
            if listLeftHalf[i] < listRightHalf[j]:
                argShuffledList[k]=listLeftHalf[i]
                i =i+1

            else:
                argShuffledList[k]=listRightHalf[j]
                j=j+1

            k=k+1

        while i < len(listLeftHalf):
            argShuffledList[k]=listLeftHalf[i]
            i=i+1
            k=k+1

        while j < len(listRightHalf):
            argShuffledList[k]=listRightHalf[j]
            j=j+1
            k=k+1

    return argShuffledList, intNumOfComp

#Perfectly learns the labels of a sample with a given query type.
#Outputs the number of queries used to learn the sample 
#(outputs a close upper bound for (comp))

#points: an array containing the sample, and a final column of all ones
#angle: a classifier of the form [normal vector, shift]
#query: either "label" or "comp," the type of query for learning.
def Perfect_Learning(points, angle, query):
	values = np.dot(points,np.transpose(angle))
	counter = 0
	uninferred = range(len(points))
	inferred = []
	indices = []
	dim = len(points[0])-1
	inf_dim=dim+1
	if query == "comp":
		stop_val = 2
	else:
		stop_val = 1
	# The core algorithm runs in this while loop and records the number of queries
	while len(uninferred) > stop_val*inf_dim:
		temp_indices = np.random.choice(uninferred,size=int(inf_dim),replace=False)
		indices.extend([temp_indices])
		inferred.extend(temp_indices)
		uninferred = list(set(uninferred).difference(set(inferred)))
		iter_infer,_,_ = infer(points,uninferred,indices,angle,query)

		if query == "comp":
			#Using merge sort, count the number of required comparisons.
			#The second term is the worst case to label all sorted points.
			counter+=MergeSort(values[temp_indices])[1]+2*np.ceil(np.log2(len(temp_indices)))
		else:
			counter+=inf_dim
		if len(iter_infer) < 1/2*len(uninferred):
			inf_dim *= 2

		inferred.extend(iter_infer)
		uninferred = list(set(uninferred).difference(set(inferred)))
	counter+=len(uninferred)
	return counter