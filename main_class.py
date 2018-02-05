from batch_and_inputs import BatchAndInputs
from train_node import TrainNode
from read_class import Read
from estimate import LocEst
import pandas as pd
import numpy as np
import json
import time

class SingleNode(object):

    def __init__(self,data,redis):

        self.__data = data

        self.__redis = redis

    def getData(self):

        return self.__data
    
    def singleNode(self):
        
        data = self.__data
        
        redis = self.__redis
        
        redis_df = pd.DataFrame(redis)
        
        data['rssi'] = data['rssi'].astype(np.int64)
        
        redis_df['spot'] = 0
        
        redis_df['spot'] = redis_df['spotId'].map( {'tyksp11': 11, 'tyksp10': 10,'tyksp09':9,'tyksp08': 8, 'tyksp03': 7,'tyksbar': 6,'tyksgrg': 5} )

        nodes = redis_df['nodeId'].unique()
 
        list_out = []
        
        json_output = []
            
        for i in range(0,len(nodes)):

            df = data[(data['nodeId']==nodes[i])]

            if df.empty==False:
            
                inputs = BatchAndInputs(df)
                
                X,y,apCount = inputs.createBatch()

                apFiltered = []
                
                apSize = np.zeros(len(apCount))
                
                for j in range(0,len(apCount)):
     
                    if apCount.iloc[j]>3:
                
                        apSize[j] = apCount.iloc[j]
                        
                        apFiltered.append(apCount.index[j])
                        
                if len(apFiltered)!=0:

                    est = LocEst(apFiltered,nodes[i],X,y,redis_df)

                    spotname,el = est.estimation()
                    
                    name = data[(data['spot']==spotname)][['spotId']]
                    
                    json_output.append({'estimation':name['spotId'].iloc[0],'nodeId':nodes[i]})
                else:
                    #print "no ap with more than 3 pings"
                    continue
            else:
                #print "empty data frame"
                continue
        
        json_data = json.dumps(json_output)

        if len(json_output)>=1:
        
            return json_data
        else:
            empty = []

            return empty
 
