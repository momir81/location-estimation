from sklearn.cluster import KMeans
import numpy as np
import scipy.misc,scipy.io,scipy.optimize,scipy.cluster.vq

class TrainNode(object):

    def __init__(self,X,y,apFiltered):
        
        self.__X = X
        
        self.__y = y
        
        self.__apFiltered = apFiltered

    def trainNode(self):
        
        X = self.__X
        
        y = self.__y
        
        apFiltered = self.__apFiltered

        max_iter = 10
        
        uniqueCats = np.unique(y[:,0])

        m,n = np.shape(X)

        Centers = []
        
        Betas = []

        idx = []
        
        totalNeuronsSize = 0
       
        for i in range(0,len(apFiltered)):

            Xc = X[y==apFiltered[i]]

            size = np.size(Xc)
            
            Xc = np.reshape(Xc,(len(Xc),-1))

            '''
            Separates data by category and clusters each category separately.
            Xc contains data from category c.
            Kmeans helps us to define clusters that represents locations and define
            their centers that further are use as mean values for calculating distancies
            from input values.
            I initilized 10 cluster per location.
            idx contains values that correpond to each cluster
            idx = [1 2 3 2 ..] means that training example with index 0 belongs to cluster 1
            and etc..
            '''

            km = KMeans(n_clusters=1)
            
            est = km.fit(Xc)
            
            centroids = est.cluster_centers_

            idx = est.predict(Xc)

            centroids = centroids[~np.isnan(centroids)]

            if idx.size == 0:
                raise NameError('Idx value is empty')
            
            numNeuron = np.size(centroids)

            #ovde moram da filtritam clustere sa 0 idx gde je nula empty
            #computes betas for each locations that contains 1 clusters/centroids
            
            betas,corr = self.computeBetas(Xc,centroids,idx)

            if corr==0:
                
                centroids=centroids.flatten()
                
                betas = betas.flatten()

                temp = np.size(centroids)
                
                totalNeuronsSize += temp

                #centers gathers all clustuers and their centroids for all categories in one array
                #centers is k*categories x 1 dimensioin
                #the same is for Betas
                centroids = centroids.tolist()

                #print 'centroids size',len(centroids)

                Centers+=centroids

                betas = betas.tolist()

                #print 'betas size',len(betas)
                
                Betas += betas
           
        #contains 3 arrays one for each class and each array contains centriods for that correponding class
        Centers = np.array(Centers)
        
        Centers = Centers.flatten()
        
        Betas = np.array(Betas)
        
        Betas = Betas.flatten()

        numRBFNeurons = totalNeuronsSize#total number after filtering nan and empty centroids

        #print "---Calculating RBF neuron activations---"

        '''
        compute RBF neuron activations for all training examples
        X_active stores RBF neuron activation for each training example: for each row in matrix X
        and one column per RBF neuron
        '''
            
        X_active = np.zeros((m,numRBFNeurons))

        for j in range(0,m):

            inputLayer = X[j,:]

            #get the activation of the neurons for each training example
            z = self.getRBFActivations(Centers,Betas,inputLayer)

            X_active[j,:] = z.T

        X_active = np.c_[np.ones(m),X_active]

        #print "--Learning output weights---"
 
        ####Here instead I can calculate theta from gradient
        ####Now it is using normal equations instead gradient

        theta = self.getTheta(numRBFNeurons,apFiltered,X_active,y)

        return Centers,Betas,theta,numRBFNeurons
        
            
       
    def computeBetas(self,Xc,centroids,idx):
        
        numNeurons = np.size(centroids)
        
        corr=0
        #corr is very import parameter. It makes sure that centers and betas are the same shape
        #in case number of clusters are more than 1 and if indices have same value like centroids
        #centroids size is identical to number of clusters. If I use 2 clusters centroids shape is 2
        #then it could happen that centroids array have identical values [[-100.00],[-100.00]]
        #then code will run only for first time beacuse the second time indices would be 0, but
        #after betas is computed new centers and betas will appended to Centers and Betas and that will
        #cause not same shape.
        #So if corr==0 everything is ok, if corr==1 do not append center and beta

        centroids = np.reshape(centroids,(len(centroids),-1))

        temp1 = centroids==centroids[0]
        
        if temp1.all()==True and np.size(temp1)>1:
            
            numNeurons = np.size(centroids)-1
            
            corr=1
        
        Xc = np.reshape(Xc,(len(Xc),-1))

        dist = []
        
        sigmas = []
   
        ####Here I can use diferent way of computing beta based on article vaznoo2.pdf

        for i in range(0,numNeurons):
            
            center = centroids[i]#indices of array starts from 0

            indices = Xc[idx==i]#idx values are from 1 to 10

            #sometimes are all indices zero and if there is no value in sample
            #that matches other centroids and that is why sample has to be big
            #enough otherwise sigma zero error raises
            temp = indices==center
            if temp.all()==True:
                
                center[0] = center[0]+0.1

            if indices.size > 0:
        
                sqrd = np.linalg.norm(indices-center)

                dist.append(sqrd)

                distArr = np.array(dist)
        
                sigmas.append(np.mean(distArr))

                #if the diff is zero then I assign some value which
                #does not affect inference and that is how I prevent
                #raising excpetion                
        
        sigmasArr = np.array(sigmas)
        
        if(np.any(sigmasArr==0)):
            
            raise NameError("One of the sigma values is zero!")
           
        betas = 1/(2*sigmasArr**2)

        return betas,corr

     
    def getRBFActivations(self,centers,betas,inputLayer):

        c = np.shape(centers)[0]
        
        betas = np.reshape(betas,(len(betas),-1))
        
        #for 3 categories/locations centers are 3 dim array and each array has k=10 (centroidSize) elements

        z = []
        ###the function calculates rbf neurons exp(-beta*(||x-center||**2))
        #### here also I can normalize every neuron (sum in normalizing vector is going over all clusters
        ###per location, if k=10 for one location sum is over 10 and etc.
        
        for i in range(0,c):
            
            centroidSize = np.size(centers[i])

            for j in range(centroidSize):

                diff = np.subtract(centers[i],inputLayer)

                #print 'diff',diff

                sqrd = np.sum(diff**2)

                z.append(np.exp(-betas[i][j]*sqrd))

            '''
             no need to normalize beacuse I am
             using normal equations
             
            '''               
        z = np.array(z)

        return z


    def getTheta(self,numRBFNeurons,apFiltered,X,y):

        theta = np.zeros(((numRBFNeurons+1),len(apFiltered)))

        for i in range(len(apFiltered)):

            y_c = y==apFiltered[i]

            part1 = np.linalg.pinv(X.T.dot(X))
           
            part2 = X.T.dot( (y_c))
 
            theta[:,i] = part2.T.dot(part1)
            
            #calculates weights using inverse(X_active)*y_c
            #X_active consists of activation of hidden node j for input i,
            #y_c is target output for category c
            
        return theta

    def evaluateRBFN(self,centres,betas,theta,inputLayer):
        # find the activations for each node

        out = self.getRBFActivations(centres,betas,inputLayer)

        m = np.shape(out)

        out = np.reshape(out,(len(out),-1))

        out = np.insert(out,0,1)

        out = np.reshape(out,(len(out),-1))
        #calcualte estimated output with gained theta 

        z = theta.T.dot(out)

        return z


    def predictOne(self,Centers,Betas,theta,pred_num,apCount):

        output = self.evaluateRBFN(Centers,Betas,theta,pred_num)
        
        indmax = np.argmax(output,axis=0)

        location = apCount[indmax[0]]

        return location


