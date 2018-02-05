import numpy as np
import pandas as pd
#from main_class import SingleNode
from train_node import TrainNode
from batch_and_inputs import BatchAndInputs
import time
import datetime

class LocEst(object):

    def __init__(self,apCount,node,X,y,redis_df):

        self.__apCount = apCount
        self.__node = node
        self.__X = X
        self.__y = y
        self.__redis = redis_df
    
    def estimation(self):

        node = self.__node

        redis = self.__redis
   
        eventData = redis[redis['nodeId']==node][['nodeId','spotId','rssi','timestamp','spot']]
        
        eventData['timestamp'] = pd.to_datetime(eventData['timestamp'])
        
        eventData['rssi'] = eventData['rssi'].astype(np.int64)
        
        eventCountAP = eventData['spotId'].value_counts()

        output,el = self.eventAPS(eventCountAP,eventData)
        
        return output,el

    def eventAPS(self,eventCountAP,eventData):

        apCount = self.__apCount
        
        X = self.__X
        
        y = self.__y

        train = TrainNode(X,y,apCount)

        Centers,Betas,theta = train.trainNode()
        
        eventMean = []

        eventSpots = eventData['spot'].value_counts()

        mean_diff = {}
        
        for f in range(0,len(eventCountAP)):
            
            eventNode = eventData[(eventData['spotId']==eventCountAP.index[f])][['rssi','spot']]

            eventRSSI = eventNode['rssi'].values

            eventMean.append(np.mean(eventRSSI))

        maxEl = max(eventMean)

        output = train.predictOne(Centers,Betas,theta,maxEl,apCount)

        return output,maxEl
