from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import TRANSACTIONS_CLEANED_PATH
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
class Forecasting():
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        self.forecasted_data = None
    def adf_test(self, column_name):
        series= self.data[column_name].dropna()
        result=adfuller(series)
        if result[1] <= 0.05:
            print("Data is stationary")
        else:
            print("Data is not stationary, differencing may be required")
        print(f"ADF Statistic: {result[0]}")
    def detect_irregular_data(self):
        tim_diff=self.data.index.to_series().diff().dt.days
        print(tim_diff.value_counts())
    def missing_date(self):
        dates = pd.date_range(start=self.data.index.min(), end=self.data.index.max())
        missing_dates = dates.difference(self.data.index)
        print(missing_dates)  
    def solve_date_in_number_cols(self, column_name):
        self.data.index=pd.to_datetime(self.data.index)
        df_resample_number=self.data[column_name].resample('D').mean()
        df_resample_number.interpolate(method='time', inplace=True)
        print(df_resample_number.isnull().sum())
        print(df_resample_number.index.to_series().diff().value_counts())
    def solve_date_in_string_cols(self, column_names):
        self.data.index = pd.to_datetime(self.data.index)

        if isinstance(column_names, str):
            column_names = [column_names]

        non_numeric_cols = self.data[column_names].copy()
        before_fill = non_numeric_cols.copy()

        for col in non_numeric_cols.columns:
            non_numeric_cols[col] = non_numeric_cols.groupby(non_numeric_cols.index.to_period('D'))[col]\
                .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown'))

        filled_changes = (before_fill != non_numeric_cols) & before_fill.isna()
        filled_data = non_numeric_cols[filled_changes]

        print("✅ Filled values:")
        print(filled_data.dropna(how='all'))

        print("\nMissing values after fill:")
        print(non_numeric_cols.isnull().sum())

        print("\nIndex frequency check:")
        print(non_numeric_cols.index.to_series().diff().value_counts())
        return non_numeric_cols
    def check_decompization(self,column_name):
        series=self.data[column_name]
        decompization=seasonal_decompose(series,model="additive",period=30)
        decompization.plot()
        plt.tight_layout()
        plt.show()
    def sarima_model(self, column_name, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        model = SARIMAX(self.data[column_name], order=order, seasonal_order=seasonal_order)
        self.forecasted_data = model.fit(disp=False)
        print(self.forecasted_data.summary())
        return self.forecasted_data
    
    def forecast_future(self, column_name, steps=30):
        if self.forecasted_data is None:
            raise ValueError("SARIMA model has not been fitted yet. Call sarima_model() first.")
        
        forecast = self.forecasted_data.get_forecast(steps=steps)
        forecast_index = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
        forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data[column_name], label='Historical Data', color='blue')
        plt.plot(forecast_series, label='Forecast', color='red')
        plt.fill_between(forecast_index, 
                         forecast.conf_int()['lower Amount'], 
                         forecast.conf_int()['upper Amount'], 
                         color='pink', alpha=0.3)
        plt.title('Future Forecast')
        plt.xlabel('Date')
        plt.ylabel(column_name)
        plt.legend()
        plt.show()
        
        
        return forecast_series 
    
              
    
    
forecasting=Forecasting(TRANSACTIONS_CLEANED_PATH)
forecasting.adf_test('Amount')
forecasting.detect_irregular_data()     
forecasting.missing_date()  
forecasting.solve_date_in_number_cols('Amount') 
forecasting.solve_date_in_string_cols(['Category','Description','Transaction Type','Account Name'])
forecasting.check_decompization('Amount')
forecasting.sarima_model('Amount', order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
forecasting.forecast_future('Amount')