from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent.parent

# Define data paths
DATA_DIR = ROOT_DIR / "data"
BUDGET_PATH = DATA_DIR / "Budget.csv"
TRANSACTIONS_PATH = DATA_DIR / "personal_transactions.csv"
TRANSACTIONS_CLEANED_PATH = DATA_DIR / "personal_transactions_cleaned.csv"

# Function to get data file path
def get_data_file_path(filename: str) -> Path:
    """
    Get the full path for a data file.
    
    Args:
        filename (str): Name of the file in the data directory
        
    Returns:
        Path: Full path to the data file
    """
    return DATA_DIR / filename 