import pandas as pd

class Analysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])

    def summary_statistics(self):
        return self.df.describe(include='all')

    def monthly_spending(self):
        monthly = self.df.groupby(self.df['Date'].dt.to_period('M')).sum(numeric_only=True)
        monthly.index = monthly.index.to_timestamp()
        return monthly

    def category_count(self):
        return self.df['Category'].value_counts()

    def category_spending(self):
        return self.df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    def transaction_type_count(self):
        return self.df['Transaction Type'].value_counts()
    def transaction_type_category(self):
        return self.df.groupby(['Transaction Type', 'Category'])['Amount'].sum().unstack(fill_value=0)
    
