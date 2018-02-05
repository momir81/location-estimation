import sys
from main_class import SingleNode
import json
from read_class import Read
#import pandas as pd


def main():

    redis = (sys.argv[1])
    
    json_data = [json.loads(line) for line in open(redis)]
    
    read = Read()

    data = read.readData()
    
    rb = SingleNode(data,json_data[0])
    
    node = rb.singleNode()
    
    


if __name__ == '__main__':
	main()
