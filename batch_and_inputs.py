import pandas as pd
import numpy as np
import time
import datetime

class BatchAndInputs(object):

    def __init__(self,data):

        self.__data = data
        
    def createBatch(self):
        
        data = self.__data

        spotCount = data['spotId'].value_counts()

        spotCount = spotCount[spotCount>=1]

        apCount = data['spot'].value_counts()

        X,y = self.createMatricies(spotCount,data)

        return X,y,apCount

    def createMatricies(self,spotCount,df1):
        
        total = 0
        
        for j in range(0,len(spotCount)):
            
            total += spotCount[spotCount.index[j]]
         
        listX = []
        
        listy = []

        X = np.zeros((total))
        
        y = np.zeros((total))
        
        temp = 0

        for j in range(0,len(spotCount)):
           
            node = df1[(df1['spotId']==spotCount.index[j])][['rssi','spot']]
            
            rssi = node['rssi'].values
            
            spots = node['spot'].values

            listX.append(rssi)
            
            listy.append(spots)
            
            X[temp:temp+len(listX[j])] = listX[j]

            y[temp:temp+len(listy[j])] = listy[j]
           
            temp += len(listX[j])
        
        X = np.reshape(X,(len(X),-1))

        y = np.reshape(y,(len(y),-1))

        return X,y
