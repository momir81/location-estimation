#from train_node import TrainNode
#import sys
from pymongo import MongoClient
import pandas as pd
from datetime import datetime,timedelta
import datetime
import numpy as np

class Read(object):
    
    def __init__(self):

        pass
 
    def readData(self):
        
        con = MongoClient('ec2-52-16-155-144.eu-west-1.compute.amazonaws.com',27020)
        
        con.rpark.authenticate('xxx','xxxxxxxxxxxxx')

        db = con.rpark
        
        start = datetime.datetime.utcnow()
        
        end = start-timedelta(weeks=4)

        #start = datetime(2016, 03, 14, 07, 00, 00)
        #end = datetime(2016, 03, 14, 12, 20, 00)

        coll = db.archivePing.find({"timestamp":{"$gte":end,\
                                            "$lt":start}})
        
        column = ['id','class','nodeId','spotId','rssi','timestamp']
 
        data = pd.DataFrame(list(coll),columns = column)
        
        data['spot'] = 0
        
        data['spot'] = data['spotId'].map( {'tyksp11': 11, 'tyksp10': 10,'tyksp09':9,'tyksp08': 8, 'tyksp03': 7,'tyksbar': 6,'tyksgrg': 5} )
        
        data = data[['nodeId','spotId','spot','rssi','timestamp']]

        return data
