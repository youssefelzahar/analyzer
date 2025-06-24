import pandas as pd
from config import TRANSACTIONS_PATH

class CheckData():
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        
    def check_data(self):
        print(self.data.head())
        print(self.data.describe())
        print(self.data.info())
        print(self.data.isnull().sum())
        print(self.data.duplicated().sum())
        
        print(self.data["Category"].value_counts())
        print(self.data["Amount"].skew())

check_data = CheckData(TRANSACTIONS_PATH)
check_data.check_data()
        