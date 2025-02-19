import os
import sys
import pandas as pd
import json
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from dotenv import load_dotenv
load_dotenv()

MONGODB_URL = os.getenv('MONGODB_URL')

import certifi
ca = certifi.where()
import pymongo

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def csv_to_json_converter(self, file_path):
        try:

            data=pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)

            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def insert_data_mongodb(self, records, database, collection):
        try:
            self.records=records
            self.database=database
            self.collection=collection

            self.mongo_client = pymongo.MongoClient(MONGODB_URL)

            self.database=self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            
            return (len(self.records))
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

if __name__=='__main__':
    FILE_PATH = 'Network_Data\phisingData.csv'
    DATABASE='ASGHARDB'
    Collection='NetworkData'
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_converter(FILE_PATH)
    print(records)
    no_of_records=networkobj.insert_data_mongodb(records, DATABASE,Collection)
    print(no_of_records)