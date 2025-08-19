# ===================================================================
# TELCO DATA SOURCE MODULE - IBM TELCO CUSTOMER CHURN DATASET
# ===================================================================
# This module specifically handles IBM's Telco Customer Churn dataset
# Source: IBM Watson Analytics Sample Data
# URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
# 
# This is REAL company data (anonymized) from a telecommunications company
# The dataset contains 7,043 customers with 21 features each
# Each customer record represents a telecommunications service subscriber
# ===================================================================

import pandas as pd  # Main library for data manipulation and analysis
import numpy as np   # Library for numerical operations and array handling
import requests      # Library for making HTTP requests to download data
import logging       # Library for logging events and debugging information
import os           # Library for operating system interface (file operations)
from typing import Dict, List, Optional, Tuple  # Type hints for better code clarity
from datetime import datetime  # For timestamp operations
from pathlib import Path      # Modern way to handle file paths
import asyncio      # For asynchronous programming (non-blocking operations)

# Set up logging to track what our code is doing
# This helps us debug issues and monitor the data ingestion process
logging.basicConfig(
    level=logging.INFO,  # Show INFO level messages and above (INFO, WARNING, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Format: timestamp - logger_name - level - message
)
logger = logging.getLogger(__name__)  # Create a logger specific to this module

class TelcoDataSource:
    """
    A specialized class for handling IBM Telco Customer Churn dataset
    
    This class provides methods to:
    1. Download the dataset from IBM's official repository
    2. Clean and preprocess the data for our ML pipeline
    3. Map telco-specific fields to our standard CustomerFeatures format
    4. Validate data quality and handle missing values
    5. Generate sample data that matches real telco patterns
    
    Why this class exists:
    - The IBM dataset has specific field names that need mapping
    - Telco data has unique characteristics (contract types, services, etc.)
    - We need to handle data quality issues specific to telecom industry
    - This provides a clean interface between raw data and our ML pipeline
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the Telco data source
        
        Args:
            data_dir (str): Directory where we'll store the downloaded dataset
                           Default is "data" folder in current directory
        
        What this method does:
        1. Sets up the data directory path
        2. Defines the URL where we'll download the dataset from
        3. Sets up field mappings between IBM dataset and our format
        4. Initializes logging for this data source
        """
        # Create a Path object for the data directory
        # Path is better than string concatenation for file paths
        self.data_dir = Path(data_dir)
        
        # URL to IBM's official Telco dataset on GitHub
        # This is a direct link to the CSV file
        self.dataset_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        
        # Local filename where we'll save the downloaded dataset
        self.dataset_filename = "telco_customer_churn.csv"
        
        # Complete path to the local dataset file
        self.dataset_path = self.data_dir / self.dataset_filename
        
        # Field mapping: IBM dataset field name -> Our standard field name
        # This dictionary maps the original column names to our standardized names
        self.field_mapping = {
            # Customer identification
            'customerID': 'customer_id',           # Unique customer identifier
            
            # Demographic information
            'gender': 'gender',                    # Customer gender (Male/Female)
            'SeniorCitizen': 'senior_citizen',     # Whether customer is senior (0/1)
            'Partner': 'has_partner',              # Has partner (Yes/No)
            'Dependents': 'has_dependents',        # Has dependents (Yes/No)
            
            # Account information
            'tenure': 'tenure_months',             # Months as customer
            'Contract': 'contract_type',           # Contract type (Month-to-month, One year, Two year)
            'PaperlessBilling': 'paperless_billing', # Uses paperless billing (Yes/No)
            'PaymentMethod': 'payment_method',     # Payment method
            
            # Service information
            'PhoneService': 'phone_service',       # Has phone service (Yes/No)
            'MultipleLines': 'multiple_lines',     # Has multiple phone lines
            'InternetService': 'internet_service', # Internet service type (DSL, Fiber optic, No)
            'OnlineSecurity': 'online_security',   # Has online security service
            'OnlineBackup': 'online_backup',       # Has online backup service
            'DeviceProtection': 'device_protection', # Has device protection
            'TechSupport': 'tech_support',         # Has tech support
            'StreamingTV': 'streaming_tv',         # Has streaming TV service
            'StreamingMovies': 'streaming_movies', # Has streaming movies service
            
            # Financial information
            'MonthlyCharges': 'monthly_charges',   # Monthly charges in USD
            'TotalCharges': 'total_charges',       # Total charges to date
            
            # Target variable (what we're trying to predict)
            'Churn': 'churn'                       # Whether customer churned (Yes/No)
        }
        
        # Log that we've initialized the data source
        logger.info(f"Initialized Telco data source with data directory: {self.data_dir}")
        logger.info(f"Dataset will be downloaded from: {self.dataset_url}")
    
    def ensure_data_directory(self) -> None:
        """
        Ensure the data directory exists, create it if it doesn't
        
        What this method does:
        1. Check if the data directory exists
        2. If not, create it (including any parent directories)
        3. Log the action taken
        
        Why this is important:
        - We need a place to store the downloaded dataset
        - The directory might not exist on first run
        - Better to check and create than to assume it exists
        """
        # Check if the directory doesn't exist
        if not self.data_dir.exists():
            # Create the directory (parents=True creates parent dirs if needed)
            # exist_ok=True means don't error if directory already exists
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {self.data_dir}")
        else:
            logger.info(f"Data directory already exists: {self.data_dir}")
    
    async def download_dataset(self, force_download: bool = False) -> bool:
        """
        Download the IBM Telco dataset from the official repository
        
        Args:
            force_download (bool): If True, download even if file already exists
                                 If False, skip download if file exists
        
        Returns:
            bool: True if download was successful, False otherwise
        
        What this method does:
        1. Check if we already have the dataset locally
        2. If not (or if forced), download it from IBM's repository
        3. Save it to our local data directory
        4. Validate that the download was successful
        5. Log all actions and any errors
        
        Why this is async:
        - Downloading files can take time (network operations)
        - Async allows other parts of our program to continue running
        - Better user experience and performance
        """
        try:
            # Ensure we have a directory to store the data
            self.ensure_data_directory()
            
            # Check if we already have the dataset and don't need to force download
            if self.dataset_path.exists() and not force_download:
                logger.info(f"Dataset already exists at {self.dataset_path}, skipping download")
                return True
            
            # Log that we're starting the download
            logger.info(f"Downloading Telco dataset from {self.dataset_url}")
            
            # Make an HTTP GET request to download the dataset
            # This gets the entire CSV file from IBM's GitHub repository
            response = requests.get(self.dataset_url)
            
            # Check if the request was successful (status code 200 means OK)
            if response.status_code == 200:
                # Write the downloaded content to our local file
                # 'wb' means write in binary mode (for non-text files)
                with open(self.dataset_path, 'wb') as file:
                    file.write(response.content)  # Write the actual data
                
                # Log successful download and file size
                file_size = self.dataset_path.stat().st_size  # Get file size in bytes
                logger.info(f"Successfully downloaded dataset: {file_size} bytes")
                return True
            else:
                # Log error if download failed
                logger.error(f"Failed to download dataset. HTTP status: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            # Handle network-related errors (no internet, server down, etc.)
            logger.error(f"Network error while downloading dataset: {e}")
            return False
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Unexpected error while downloading dataset: {e}")
            return False
    
    def load_raw_data(self) -> Optional[pd.DataFrame]:
        """
        Load the raw Telco dataset from local file
        
        Returns:
            pd.DataFrame: The raw dataset as a pandas DataFrame
            None: If loading failed
        
        What this method does:
        1. Check if the dataset file exists locally
        2. Load it using pandas (which handles CSV parsing)
        3. Log information about the loaded dataset
        4. Return the DataFrame or None if loading failed
        
        Why we use pandas:
        - Excellent for handling CSV files and tabular data
        - Provides powerful data manipulation capabilities
        - Industry standard for data science work
        """
        try:
            # Check if the dataset file exists
            if not self.dataset_path.exists():
                logger.error(f"Dataset file not found: {self.dataset_path}")
                logger.info("Try running download_dataset() first")
                return None
            
            # Load the CSV file into a pandas DataFrame
            # pandas automatically handles headers, data types, etc.
            logger.info(f"Loading dataset from {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
            
            # Log information about the loaded dataset
            logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Dataset columns: {list(df.columns)}")
            
            return df
            
        except pd.errors.EmptyDataError:
            # Handle case where CSV file is empty
            logger.error("Dataset file is empty")
            return None
        except pd.errors.ParserError as e:
            # Handle case where CSV file is malformed
            logger.error(f"Error parsing CSV file: {e}")
            return None
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Unexpected error loading dataset: {e}")
            return None
    
    def clean_and_preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the raw Telco data
        
        Args:
            df (pd.DataFrame): Raw dataset from IBM
        
        Returns:
            pd.DataFrame: Cleaned and preprocessed dataset
        
        What this method does:
        1. Handle missing values (empty strings, spaces, etc.)
        2. Convert data types to appropriate formats
        3. Standardize categorical values (Yes/No, Male/Female, etc.)
        4. Create derived features (like age from senior citizen flag)
        5. Apply field mappings to standardize column names
        6. Log all transformations for transparency
        
        Why cleaning is important:
        - Real data is messy and has inconsistencies
        - ML models need clean, consistent data formats
        - Missing values can break model training
        - Standardized formats make the data pipeline robust
        """
        logger.info("Starting data cleaning and preprocessing")
        
        # Create a copy to avoid modifying the original DataFrame
        # This is a best practice to preserve the original data
        cleaned_df = df.copy()
        
        # Step 1: Handle the TotalCharges column
        # Problem: TotalCharges is stored as string but should be numeric
        # Some values are empty strings (new customers with no charges yet)
        logger.info("Cleaning TotalCharges column...")
        
        # Replace empty strings and spaces with NaN (Not a Number)
        # This makes them easy to identify and handle
        cleaned_df['TotalCharges'] = cleaned_df['TotalCharges'].replace([' ', ''], np.nan)
        
        # Convert to numeric, forcing errors to become NaN
        # errors='coerce' means invalid values become NaN instead of raising errors
        cleaned_df['TotalCharges'] = pd.to_numeric(cleaned_df['TotalCharges'], errors='coerce')
        
        # For customers with NaN total charges, set to 0 (they're new customers)
        # This is a business logic decision based on domain knowledge
        total_charges_nan_count = cleaned_df['TotalCharges'].isna().sum()
        logger.info(f"Found {total_charges_nan_count} customers with missing TotalCharges, setting to 0")
        cleaned_df['TotalCharges'] = cleaned_df['TotalCharges'].fillna(0)
        
        # Step 2: Standardize Yes/No values to boolean
        # Many columns use "Yes"/"No" strings, but booleans are better for ML
        yes_no_columns = [
            'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'
        ]
        
        logger.info(f"Converting Yes/No columns to boolean: {yes_no_columns}")
        for col in yes_no_columns:
            if col in cleaned_df.columns:
                # Map "Yes" to True, "No" to False
                cleaned_df[col] = cleaned_df[col].map({'Yes': True, 'No': False})
        
        # Step 3: Handle service columns that can have multiple values
        # Some columns have values like "Yes", "No", "No internet service"
        service_columns = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        logger.info(f"Processing service columns: {service_columns}")
        for col in service_columns:
            if col in cleaned_df.columns:
                # Map values: "Yes" -> True, everything else -> False
                # This treats "No" and "No internet service" the same way
                cleaned_df[col] = cleaned_df[col].map(lambda x: True if x == 'Yes' else False)
        
        # Step 4: Handle MultipleLines column
        # Values: "Yes", "No", "No phone service"
        if 'MultipleLines' in cleaned_df.columns:
            logger.info("Processing MultipleLines column")
            cleaned_df['MultipleLines'] = cleaned_df['MultipleLines'].map({
                'Yes': True,
                'No': False,
                'No phone service': False  # No phone service means no multiple lines
            })
        
        # Step 5: Standardize gender values
        # Convert to lowercase for consistency
        if 'gender' in cleaned_df.columns:
            logger.info("Standardizing gender values")
            cleaned_df['gender'] = cleaned_df['gender'].str.lower()
        
        # Step 6: Create age approximation from SeniorCitizen
        # SeniorCitizen is 1 for seniors (65+), 0 for non-seniors
        # We'll estimate age for ML model compatibility
        if 'SeniorCitizen' in cleaned_df.columns:
            logger.info("Creating age approximation from SeniorCitizen flag")
            # Seniors: age 65-75 (random within range)
            # Non-seniors: age 25-64 (random within range)
            np.random.seed(42)  # For reproducible results
            
            senior_mask = cleaned_df['SeniorCitizen'] == 1
            cleaned_df['age'] = np.where(
                senior_mask,
                np.random.randint(65, 76, size=len(cleaned_df)),  # Seniors: 65-75
                np.random.randint(25, 65, size=len(cleaned_df))   # Non-seniors: 25-64
            )
        
        # Step 7: Create income approximation based on monthly charges
        # Higher monthly charges often correlate with higher income
        if 'MonthlyCharges' in cleaned_df.columns:
            logger.info("Creating income approximation based on monthly charges")
            # Income estimate: monthly_charges * 600 + random variation
            # This gives realistic income ranges for different service levels
            np.random.seed(43)  # Different seed for income
            base_income = cleaned_df['MonthlyCharges'] * 600
            income_variation = np.random.normal(0, 15000, len(cleaned_df))  # Add some randomness
            cleaned_df['income'] = np.maximum(20000, base_income + income_variation)  # Minimum $20k
        
        # Step 8: Apply field mappings to standardize column names
        # This maps IBM's column names to our standard format
        logger.info("Applying field mappings to standardize column names")
        
        # Only rename columns that exist in both the DataFrame and our mapping
        columns_to_rename = {k: v for k, v in self.field_mapping.items() if k in cleaned_df.columns}
        cleaned_df = cleaned_df.rename(columns=columns_to_rename)
        
        logger.info(f"Renamed {len(columns_to_rename)} columns")
        logger.info(f"Final dataset shape: {cleaned_df.shape}")
        
        return cleaned_df
    
    def convert_to_standard_format(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert cleaned DataFrame to our standard customer features format
        
        Args:
            df (pd.DataFrame): Cleaned and preprocessed DataFrame
        
        Returns:
            List[Dict]: List of customer records in standard format
        
        What this method does:
        1. Convert DataFrame rows to dictionaries
        2. Ensure all required fields are present
        3. Fill in missing fields with default values
        4. Validate data types and ranges
        5. Add metadata (timestamp, data source)
        6. Log conversion statistics
        
        Why this conversion is needed:
        - Our ML pipeline expects a specific data format
        - Different data sources have different formats
        - This creates a consistent interface for all data sources
        - Makes the pipeline more robust and maintainable
        """
        logger.info("Converting to standard customer features format")
        
        # Convert DataFrame to list of dictionaries
        # Each row becomes a dictionary with column names as keys
        records = df.to_dict('records')
        
        # Standard format requirements for our ML pipeline
        standard_records = []
        
        for i, record in enumerate(records):
            try:
                # Create a standardized customer record
                standard_record = {
                    # Required fields with fallback defaults
                    'customer_id': record.get('customer_id', f'TELCO_{i:06d}'),
                    'age': int(record.get('age', 35)),  # Default age if missing
                    'gender': str(record.get('gender', 'unknown')).lower(),
                    'income': float(record.get('income', 50000)),  # Default income
                    'tenure_months': int(record.get('tenure_months', 0)),
                    'monthly_charges': float(record.get('monthly_charges', 0)),
                    'total_charges': float(record.get('total_charges', 0)),
                    
                    # Service features with boolean conversion
                    'phone_service': bool(record.get('phone_service', False)),
                    'internet_service': self._standardize_internet_service(
                        record.get('internet_service', 'No')
                    ),
                    'streaming_tv': bool(record.get('streaming_tv', False)),
                    'streaming_movies': bool(record.get('streaming_movies', False)),
                    'tech_support': bool(record.get('tech_support', False)),
                    'device_protection': bool(record.get('device_protection', False)),
                    
                    # Contract and billing features
                    'contract_type': self._standardize_contract_type(
                        record.get('contract_type', 'Month-to-month')
                    ),
                    'paperless_billing': bool(record.get('paperless_billing', False)),
                    'payment_method': self._standardize_payment_method(
                        record.get('payment_method', 'Electronic check')
                    ),
                    
                    # Metadata
                    'timestamp': datetime.now(),
                    'data_source': 'ibm_telco_dataset',
                    
                    # Additional telco-specific fields
                    'has_partner': bool(record.get('has_partner', False)),
                    'has_dependents': bool(record.get('has_dependents', False)),
                    'senior_citizen': bool(record.get('senior_citizen', False)),
                    'multiple_lines': bool(record.get('multiple_lines', False)),
                    'online_security': bool(record.get('online_security', False)),
                    'online_backup': bool(record.get('online_backup', False)),
                    
                    # Target variable (if available)
                    'churn': self._standardize_churn_value(record.get('churn'))
                }
                
                standard_records.append(standard_record)
                
            except Exception as e:
                # Log errors but continue processing other records
                logger.warning(f"Error processing record {i}: {e}")
                continue
        
        logger.info(f"Successfully converted {len(standard_records)} out of {len(records)} records")
        
        return standard_records
    
    def _standardize_internet_service(self, value: str) -> str:
        """
        Standardize internet service values to match our expected format
        
        Args:
            value (str): Original internet service value
        
        Returns:
            str: Standardized internet service value
        
        What this does:
        - Maps various formats to our standard: "DSL", "Fiber optic", "No"
        - Handles case sensitivity and variations
        """
        if pd.isna(value) or value.lower() in ['no', 'none']:
            return 'No'
        elif 'dsl' in value.lower():
            return 'DSL'
        elif 'fiber' in value.lower():
            return 'Fiber optic'
        else:
            return 'No'  # Default fallback
    
    def _standardize_contract_type(self, value: str) -> str:
        """
        Standardize contract type values to match our expected format
        
        Args:
            value (str): Original contract type value
        
        Returns:
            str: Standardized contract type value
        """
        if pd.isna(value):
            return 'Month-to-month'  # Default
        
        value_lower = value.lower()
        if 'month' in value_lower:
            return 'Month-to-month'
        elif 'one year' in value_lower or '1 year' in value_lower:
            return 'One year'
        elif 'two year' in value_lower or '2 year' in value_lower:
            return 'Two year'
        else:
            return 'Month-to-month'  # Default fallback
    
    def _standardize_payment_method(self, value: str) -> str:
        """
        Standardize payment method values to match our expected format
        
        Args:
            value (str): Original payment method value
        
        Returns:
            str: Standardized payment method value
        """
        if pd.isna(value):
            return 'Electronic check'  # Default
        
        value_lower = value.lower()
        if 'electronic' in value_lower or 'auto' in value_lower:
            return 'Electronic check'
        elif 'mail' in value_lower:
            return 'Mailed check'
        elif 'bank' in value_lower or 'transfer' in value_lower:
            return 'Bank transfer'
        elif 'credit' in value_lower or 'card' in value_lower:
            return 'Credit card'
        else:
            return 'Electronic check'  # Default fallback
    
    def _standardize_churn_value(self, value) -> Optional[bool]:
        """
        Standardize churn values to boolean format
        
        Args:
            value: Original churn value (could be "Yes"/"No", 1/0, True/False)
        
        Returns:
            bool or None: True if churned, False if not churned, None if unknown
        """
        if pd.isna(value):
            return None
        
        if isinstance(value, str):
            return value.lower() == 'yes'
        elif isinstance(value, (int, float)):
            return bool(value)
        elif isinstance(value, bool):
            return value
        else:
            return None
    
    async def get_sample_customers(self, count: int = 100) -> List[Dict]:
        """
        Get a sample of customers from the Telco dataset
        
        Args:
            count (int): Number of customers to return (default 100)
        
        Returns:
            List[Dict]: List of customer records in standard format
        
        What this method does:
        1. Download dataset if not available locally
        2. Load and clean the data
        3. Convert to standard format
        4. Return a random sample of the requested size
        5. Log all operations for transparency
        
        This is the main entry point for getting Telco customer data
        """
        try:
            logger.info(f"Getting {count} sample customers from Telco dataset")
            
            # Step 1: Ensure we have the dataset
            download_success = await self.download_dataset()
            if not download_success:
                logger.error("Failed to download dataset")
                return []
            
            # Step 2: Load the raw data
            raw_df = self.load_raw_data()
            if raw_df is None:
                logger.error("Failed to load dataset")
                return []
            
            # Step 3: Clean and preprocess the data
            cleaned_df = self.clean_and_preprocess_data(raw_df)
            
            # Step 4: Convert to standard format
            standard_records = self.convert_to_standard_format(cleaned_df)
            
            # Step 5: Return a random sample
            if len(standard_records) > count:
                # Use numpy for reproducible random sampling
                np.random.seed(44)  # For reproducible results
                sample_indices = np.random.choice(len(standard_records), count, replace=False)
                sample_records = [standard_records[i] for i in sample_indices]
                logger.info(f"Returning random sample of {count} customers")
            else:
                sample_records = standard_records
                logger.info(f"Returning all {len(sample_records)} available customers")
            
            return sample_records
            
        except Exception as e:
            logger.error(f"Error getting sample customers: {e}")
            return []
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the Telco dataset
        
        Returns:
            Dict: Information about the dataset including size, features, etc.
        """
        try:
            raw_df = self.load_raw_data()
            if raw_df is None:
                return {"error": "Dataset not available"}
            
            # Calculate dataset statistics
            info = {
                "dataset_name": "IBM Telco Customer Churn",
                "source": "IBM Watson Analytics",
                "total_customers": len(raw_df),
                "total_features": len(raw_df.columns),
                "churn_rate": (raw_df['Churn'] == 'Yes').mean() if 'Churn' in raw_df.columns else None,
                "features": list(raw_df.columns),
                "data_types": raw_df.dtypes.to_dict(),
                "missing_values": raw_df.isnull().sum().to_dict(),
                "file_path": str(self.dataset_path),
                "file_size_bytes": self.dataset_path.stat().st_size if self.dataset_path.exists() else None
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return {"error": str(e)}

# ===================================================================
# CONVENIENCE FUNCTIONS FOR EASY USAGE
# ===================================================================

async def download_telco_dataset(data_dir: str = "data") -> bool:
    """
    Convenience function to download the Telco dataset
    
    Args:
        data_dir (str): Directory to store the dataset
    
    Returns:
        bool: True if successful, False otherwise
    """
    telco = TelcoDataSource(data_dir)
    return await telco.download_dataset()

async def get_telco_sample_data(count: int = 100, data_dir: str = "data") -> List[Dict]:
    """
    Convenience function to get sample Telco customer data
    
    Args:
        count (int): Number of customers to return
        data_dir (str): Directory where dataset is stored
    
    Returns:
        List[Dict]: List of customer records
    """
    telco = TelcoDataSource(data_dir)
    return await telco.get_sample_customers(count)

def get_telco_dataset_info(data_dir: str = "data") -> Dict:
    """
    Convenience function to get Telco dataset information
    
    Args:
        data_dir (str): Directory where dataset is stored
    
    Returns:
        Dict: Dataset information
    """
    telco = TelcoDataSource(data_dir)
    return telco.get_dataset_info()

# ===================================================================
# MAIN EXECUTION FOR TESTING
# ===================================================================

if __name__ == "__main__":
    """
    Test the Telco data source functionality
    This runs when the script is executed directly
    """
    
    async def test_telco_data_source():
        """Test all functionality of the Telco data source"""
        
        print("🔗 TELCO DATA SOURCE TEST")
        print("=" * 50)
        
        # Initialize the data source
        telco = TelcoDataSource()
        
        # Test 1: Download dataset
        print("\n1️⃣  Testing dataset download...")
        download_success = await telco.download_dataset()
        print(f"Download successful: {download_success}")
        
        # Test 2: Get dataset info
        print("\n2️⃣  Getting dataset information...")
        info = telco.get_dataset_info()
        if 'error' not in info:
            print(f"Dataset: {info['dataset_name']}")
            print(f"Total customers: {info['total_customers']:,}")
            print(f"Features: {info['total_features']}")
            print(f"Churn rate: {info['churn_rate']:.2%}")
        else:
            print(f"Error getting info: {info['error']}")
        
        # Test 3: Get sample data
        print("\n3️⃣  Getting sample customer data...")
        sample_customers = await telco.get_sample_customers(10)
        print(f"Retrieved {len(sample_customers)} sample customers")
        
        if sample_customers:
            print("\nSample customer:")
            first_customer = sample_customers[0]
            for key, value in first_customer.items():
                print(f"  {key}: {value}")
        
        print("\n✅ Telco data source test complete!")
    
    # Run the test
    asyncio.run(test_telco_data_source())
