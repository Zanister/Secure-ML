import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class PreProcessNetFlowCsv:
    """
    Class to preprocess NetFlow CSV data for network traffic analysis.
    Based on the original preprocessing logic from the provided code snippets.
    """
    def __init__(self, csv_file_name):
        """
        Initialize with a CSV file
        
        Args:
            csv_file_name (str): Path to the CSV file to process
        """
        self.csv_file_name = csv_file_name
        self.df = pd.read_csv(self.csv_file_name)
        self.drop_columns = None
        self.x = None
        self.y = None
    
    def set_drop_columns(self, drop_columns):
        """Set columns to be dropped during preprocessing"""
        self.drop_columns = drop_columns
    
    def drop_unused_column(self):
        """Drop unused columns based on the drop_columns list"""
        self.df = self.df.drop(self.drop_columns, axis=1, errors='ignore')
    
    def encode_text_dummy(self, name):
        """
        Encode text values to dummy variables
        (i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
        """
        if name in self.df.columns:
            dummies = pd.get_dummies(self.df[name])
            for x in dummies.columns:
                dummy_name = f"{name}-{x}"
                self.df[dummy_name] = dummies[x]
            self.df.drop(name, axis=1, inplace=True)
    
    def pre_process(self):
        """Handle special cases and data cleaning"""
        # Convert specific columns to float32
        if 'Flow Byts/s' in self.df.columns:
            self.df['Flow Byts/s'] = self.df['Flow Byts/s'].astype('float32')
        if 'Flow Pkts/s' in self.df.columns:
            self.df['Flow Pkts/s'] = self.df['Flow Pkts/s'].astype('float32')
        
        # Replace infinity values and drop NaN values
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.dropna()
        
        # Handle 'No Label' values
        if 'Label' in self.df.columns:
            self.df.loc[self.df['Label']=='No Label', 'Label'] = 0
            # Also handle 'NeedManualLabel' for compatibility with newer data
            self.df = self.df[~self.df['Label'].isin(['NeedManualLabel'])]
   
    def split_x_y(self, label_name):
        """
        Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
        
        Args:
            label_name (str): Name of the label/target column
            
        Returns:
            numpy.ndarray: Feature array (X)
        """
        def to_xy(df, target):
            result = []
            for x in df.columns:
                if x != target:
                    result.append(x)
            
            # Find out the type of the target column
            target_type = df[target].dtypes
            target_type = target_type[0] if hasattr(
                target_type, '__iter__') else target_type
            
            # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
            if target_type in (np.int64, np.int32):
                # Classification
                dummies = pd.get_dummies(df[target])
                return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
            
            # Regression or direct use
            return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)
        
        if label_name in self.df.columns:
            self.x, self.y = to_xy(self.df, label_name)
        else:
            # If label column doesn't exist, just use all columns as features
            self.x = self.df.values.astype(np.float32)
            self.y = None
            
        return self.x
    
    def normalize(self):
        """Normalize features using Min-Max scaling"""
        if self.x is not None:
            scaler = MinMaxScaler()
            self.x = scaler.fit_transform(self.x)
            return self.x
        return None
    
    def get_dataframe(self):
        """Return the current DataFrame"""
        return self.df