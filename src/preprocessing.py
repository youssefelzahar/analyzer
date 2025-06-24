from config import TRANSACTIONS_PATH, TRANSACTIONS_CLEANED_PATH
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Preprocessing():
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        
    def check_outliers(self):
        for col in self.data.columns:
            if self.data[col].dtype != 'object':
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                print(f"Column: {col}")
                print(f"Outliers: {len(outliers)}")
        return self.data
    
    def remove_outliers(self):
        for col in self.data.columns:
            if self.data[col].dtype != 'object':
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
                print(f"Column: {col}")
                print(f"Outliers: {len(self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)])}")
        return self.data
  
    def solve_skewness(self):
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                if (self.data[col] <= 0).any():
                    print(f"Column '{col}' contains non-positive values. Applying log1p transformation.")
                    self.data[col] = np.log1p(self.data[col])  # log(1 + x)
                else:
                    self.data[col] = np.log(self.data[col])
        return self.data

    def convert_to_datetime(self):
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        return self.data
    def encode_categorical_data(self):
        label_encoder = LabelEncoder()
        self.data["Category_encoded"]=label_encoder.fit_transform(self.data["Category"])
        return self.data

    def data_cleaning(self):
        self.check_outliers()
        self.remove_outliers()
        self.solve_skewness()
        self.convert_to_datetime()
        self.encode_categorical_data()
        return self.data

    def save_data(self, output_file_path):
        """Saves the dataframe to a CSV file."""
        self.data.to_csv(output_file_path, index=False)
        print(f"Cleaned data saved to {output_file_path}")

if __name__ == "__main__":
    preprocessing = Preprocessing(TRANSACTIONS_PATH)
    cleaned_data = preprocessing.data_cleaning()
    
    print("\n--- Cleaned Data Info ---")
    print(cleaned_data.info())
    print(f"Date column type: {cleaned_data['Date'].dtype}")
    
    preprocessing.save_data(TRANSACTIONS_CLEANED_PATH)
        
       