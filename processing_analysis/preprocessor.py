import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class PreProcessCICFlow:
    """
    A class to preprocess CSV data generated from CICFlowMeter for machine learning classification.
    Enhanced with robust debugging, error handling, and division by zero protection.
    """
    
    def __init__(self, csv_file_name):
        """
        Initialize the PreProcessCICFlow instance.
        :param csv_file_name: The name of the CICFlowMeter CSV file to preprocess.
        """
        self.csv_file_name = csv_file_name
        print(f"[DEBUG] Loading CSV file: {csv_file_name}")
        try:
            # Try with different encoding options if needed
            try:
                self.df = pd.read_csv(self.csv_file_name)
            except UnicodeDecodeError:
                # Try with different encoding if default fails
                self.df = pd.read_csv(self.csv_file_name, encoding='latin1')
                
            print(f"[DEBUG] Successfully loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Check if DataFrame is empty
            if self.df.empty:
                raise ValueError(f"The CSV file {csv_file_name} is empty or couldn't be read properly")
                
        except Exception as e:
            print(f"[ERROR] Failed to load CSV file: {str(e)}")
            raise
        
        self.drop_columns = None  # Columns to be dropped
        self.x = None  # Features
        self.y = None  # Labels
        
        # Debug information about the loaded data
        self.print_column_info()
        
        # Clean column names
        self._clean_column_names()
    
    def _clean_column_names(self):
        """
        Clean column names by stripping whitespace and replacing problematic characters.
        """
        old_columns = list(self.df.columns)
        self.df.columns = self.df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')
        new_columns = list(self.df.columns)
        
        # Print changes for debugging
        changed = [f"{old} -> {new}" for old, new in zip(old_columns, new_columns) if old != new]
        if changed:
            print(f"[DEBUG] Cleaned {len(changed)} column names:")
            for change in changed:
                print(f"  {change}")
    
    def print_column_info(self):
        """
        Print column names and a sample of data to debug preprocessing issues.
        """
        print("[DEBUG] CSV columns:", list(self.df.columns))
        if not self.df.empty:
            print("[DEBUG] First row sample:")
            sample_dict = self.df.iloc[0].to_dict()
            for k, v in sample_dict.items():
                print(f"  {k}: {v} (type: {type(v).__name__})")
            print("[DEBUG] Data types:")
            for col, dtype in self.df.dtypes.items():
                print(f"  {col}: {dtype}")
        else:
            print("[WARNING] DataFrame is empty!")
    
    def map_standard_columns(self):
        """
        Map CICFlowMeter column variations to standard names.
        This handles different column naming conventions across CICFlowMeter versions.
        """
        # Create a mapping dictionary of possible column names to standard names
        column_mapping = {
            # Flow identifiers
            'Flow_ID': 'Flow_ID',
            'Flow ID': 'Flow_ID',
            ' Flow ID': 'Flow_ID',
            'FlowID': 'Flow_ID',
            
            # IP addresses
            'Src_IP': 'Source_IP',
            'Src IP': 'Source_IP',
            ' Src IP': 'Source_IP',
            'SrcIP': 'Source_IP',
            'Source IP': 'Source_IP',
            
            'Dst_IP': 'Destination_IP',
            'Dst IP': 'Destination_IP',
            ' Dst IP': 'Destination_IP',
            'DstIP': 'Destination_IP',
            'Destination IP': 'Destination_IP',
            
            # Ports
            'Src_Port': 'Source_Port',
            'Src Port': 'Source_Port',
            ' Src Port': 'Source_Port',
            'SrcPort': 'Source_Port',
            
            'Dst_Port': 'Destination_Port',
            'Dst Port': 'Destination_Port',
            ' Dst Port': 'Destination_Port',
            'DstPort': 'Destination_Port',
            
            # Flow metrics
            'Flow_Bytes_s': 'Flow_Bytes_s',
            'Flow Bytes/s': 'Flow_Bytes_s',
            ' Flow Bytes/s': 'Flow_Bytes_s',
            
            'Flow_Packets_s': 'Flow_Packets_s',
            'Flow Packets/s': 'Flow_Packets_s',
            ' Flow Packets/s': 'Flow_Packets_s',
            
            'Packet_Length_Variance': 'Packet_Length_Variance',
            'Packet Length Variance': 'Packet_Length_Variance',
            ' Packet Length Variance': 'Packet_Length_Variance',
            
            'Avg_Packet_Size': 'Avg_Packet_Size',
            'Average Packet Size': 'Avg_Packet_Size',
            ' Average Packet Size': 'Avg_Packet_Size',
            'Avg Packet Size': 'Avg_Packet_Size',
            'Average_Packet_Size': 'Avg_Packet_Size',
            
            # Add more mappings as needed
        }
        
        # Rename columns based on the mapping
        renamed_columns = {}
        for old_col in self.df.columns:
            if old_col in column_mapping:
                renamed_columns[old_col] = column_mapping[old_col]
        
        if renamed_columns:
            print(f"[INFO] Renaming {len(renamed_columns)} columns to standard format:")
            for old, new in renamed_columns.items():
                print(f"  {old} -> {new}")
            self.df.rename(columns=renamed_columns, inplace=True)
    
    def set_drop_columns(self, drop_columns):
        """
        Set columns to be dropped.
        :param drop_columns: List of column names to drop.
        """
        self.drop_columns = drop_columns
        print(f"[DEBUG] Set columns to drop: {drop_columns}")
    
    def drop_unused_columns(self):
        """
        Drop specified columns if they exist.
        """
        if self.drop_columns:
            dropped = []
            skipped = []
            for col in self.drop_columns:
                if col in self.df.columns:
                    self.df.drop(columns=[col], inplace=True)
                    dropped.append(col)
                else:
                    skipped.append(col)
            
            if dropped:
                print(f"[INFO] Dropped {len(dropped)} columns: {dropped}")
            if skipped:
                print(f"[INFO] Skipped dropping {len(skipped)} non-existent columns: {skipped}")
    
    def preprocess(self):
        """
        Perform preprocessing:
        - Map column names to standard format
        - Convert numeric columns to float32
        - Handle missing and infinite values
        - Remove rows with missing values
        """
        # Check if DataFrame is empty
        if self.df.empty:
            print("[ERROR] DataFrame is empty. Cannot preprocess.")
            return False
            
        # First map the columns to handle naming variations
        self.map_standard_columns()
        
        # Get all numeric columns (more robust than hardcoding)
        numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        print(f"[DEBUG] Auto-detected numeric columns: {numeric_columns}")
        
        # Add known numeric columns that might not be detected automatically
        potential_numeric_cols = [
            'Flow_Bytes_s', 'Flow_Packets_s', 'Packet_Length_Variance', 
            'Avg_Packet_Size', 'Flow_Duration', 'Total_Fwd_Packets',
            'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets'
        ]
        
        converted_cols = []
        failed_cols = []
        
        for col in potential_numeric_cols:
            if col in self.df.columns and col not in numeric_columns:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    numeric_columns.append(col)
                    converted_cols.append(col)
                except Exception as e:
                    failed_cols.append(f"{col} ({str(e)})")
        
        if converted_cols:
            print(f"[INFO] Converted {len(converted_cols)} columns to numeric: {converted_cols}")
        if failed_cols:
            print(f"[WARNING] Failed to convert {len(failed_cols)} columns to numeric: {failed_cols}")
        
        # *** DIVISION BY ZERO FIX ***
        # Handle division by zero issues in flow rate calculations
        if 'Flow_Duration' in self.df.columns:
            # Make sure all durations are at least a small positive number to avoid div/0
            # Use a small value instead of zero (1 nanosecond)
            self.df['Flow_Duration'] = self.df['Flow_Duration'].replace(0, 1e-9)
            
            # Fix Flow_Bytes_s calculation if needed
            if 'Flow_Bytes_s' in self.df.columns and 'Total_Length_of_Fwd_Packets' in self.df.columns and 'Total_Length_of_Bwd_Packet' in self.df.columns:
                # Recalculate to ensure consistency
                try:
                    total_bytes = self.df['Total_Length_of_Fwd_Packets'] + self.df['Total_Length_of_Bwd_Packet']
                    self.df['Flow_Bytes_s'] = total_bytes / self.df['Flow_Duration']
                    print("[INFO] Recalculated Flow_Bytes_s to prevent division by zero")
                except Exception as e:
                    print(f"[WARNING] Could not recalculate Flow_Bytes_s: {str(e)}")
            
            # Fix Flow_Packets_s calculation if needed
            if 'Flow_Packets_s' in self.df.columns and 'Total_Fwd_Packet' in self.df.columns and 'Total_Bwd_packets' in self.df.columns:
                try:
                    total_packets = self.df['Total_Fwd_Packet'] + self.df['Total_Bwd_packets']
                    self.df['Flow_Packets_s'] = total_packets / self.df['Flow_Duration']
                    print("[INFO] Recalculated Flow_Packets_s to prevent division by zero")
                except Exception as e:
                    print(f"[WARNING] Could not recalculate Flow_Packets_s: {str(e)}")
        
        # Check for and report NaN values before cleaning
        nan_counts = self.df.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        if not cols_with_nan.empty:
            print(f"[WARNING] Found NaN values in {len(cols_with_nan)} columns before cleaning:")
            for col, count in cols_with_nan.items():
                print(f"  {col}: {count} NaN values ({(count/len(self.df))*100:.2f}%)")
        
        # Check for and report infinite values
        inf_counts = self.df.replace([np.inf, -np.inf], np.nan).isna().sum() - self.df.isna().sum()
        cols_with_inf = inf_counts[inf_counts > 0]
        if not cols_with_inf.empty:
            print(f"[WARNING] Found infinite values in {len(cols_with_inf)} columns:")
            for col, count in cols_with_inf.items():
                print(f"  {col}: {count} infinite values ({(count/len(self.df))*100:.2f}%)")
            
            # Replace infinite values with NaN
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # If we have too many NaN values (over 30% in a column), we might want to drop that column
        high_nan_cols = [col for col, count in nan_counts.items() if count/len(self.df) > 0.3]
        if high_nan_cols:
            print(f"[WARNING] Columns with >30% NaN values (consider dropping): {high_nan_cols}")
        
        # Drop rows with NaN in numeric columns
        original_count = len(self.df)
        self.df.dropna(subset=numeric_columns, inplace=True)
        rows_dropped = original_count - len(self.df)
        
        print(f"[DEBUG] Dropped {rows_dropped} rows with NaN values ({(rows_dropped/original_count)*100:.2f}% of data)")
        print(f"[DEBUG] Number of rows remaining after preprocessing: {len(self.df)}")
        
        if len(self.df) == 0:
            raise ValueError("[ERROR] Dataset is empty after preprocessing. Check input data.")
            
        return True
    
    def inspect_label_column(self, label_name):
        """
        Inspect the label column to see what values it contains.
        This helps identify issues with non-numeric labels.
        """
        if label_name not in self.df.columns:
            print(f"[ERROR] Label column '{label_name}' not found. Available columns: {list(self.df.columns)}")
            return False
        
        print(f"[INFO] Label column '{label_name}' value counts:")
        value_counts = self.df[label_name].value_counts()
        print(value_counts)
        
        print(f"[INFO] Label column '{label_name}' data type: {self.df[label_name].dtype}")
        
        # Check for problematic values
        if self.df[label_name].dtype == object:  # If string type
            unique_values = self.df[label_name].unique()
            print(f"[DEBUG] Unique values in label column: {unique_values}")
            
            # Check for specific problematic values
            problematic = [val for val in unique_values if isinstance(val, str) and val in ['NeedManualLabel', 'Undefined']]
            if problematic:
                print(f"[WARNING] Found problematic label values: {problematic}")
                
                # Count occurrences
                for val in problematic:
                    count = (self.df[label_name] == val).sum()
                    print(f"  '{val}' occurs in {count} rows ({(count/len(self.df))*100:.2f}% of data)")
        
        return True
    
    def split_x_y(self, label_name, prediction_mode=True):

        # Check if DataFrame is empty
        if self.df.empty:
            raise ValueError("[ERROR] DataFrame is empty. Cannot split features and labels.")

        # First inspect the label column
        if not self.inspect_label_column(label_name):
            raise ValueError(f"[ERROR] Cannot process label column '{label_name}'")
    
        if label_name not in self.df.columns:
            raise ValueError(f"[ERROR] Label column '{label_name}' not found in the dataset.")
    
        # Handle non-numeric labels
        if self.df[label_name].dtype == object:  # Check if label is string type
            print(f"[INFO] Found non-numeric labels in {label_name}. Handling special cases...")
        
            # Check percentage of 'NeedManualLabel' values
            if 'NeedManualLabel' in self.df[label_name].values:
                need_manual_count = (self.df[label_name] == 'NeedManualLabel').sum()
                need_manual_percentage = (need_manual_count / len(self.df)) * 100
            
                if need_manual_percentage == 100 and prediction_mode:
                    # If we're in prediction mode and all rows need labels, 
                    # assign a placeholder label instead of filtering
                    print(f"[INFO] All rows ({need_manual_count}) have 'NeedManualLabel'. Using placeholder labels for prediction mode.")
                    self.df[label_name] = 0  # Using 0 as a placeholder label
                elif need_manual_percentage >= 50 and prediction_mode:
                    # If more than half the data needs labels and we're in prediction mode
                    print(f"[INFO] {need_manual_count} rows ({need_manual_percentage:.2f}%) have 'NeedManualLabel'. Using placeholder labels for these rows.")
                    self.df.loc[self.df[label_name] == 'NeedManualLabel', label_name] = 0
                elif prediction_mode:
                    # If in prediction mode but only a small portion needs labels
                    print(f"[INFO] {need_manual_count} rows ({need_manual_percentage:.2f}%) have 'NeedManualLabel'. Using placeholder labels for these rows.")
                    self.df.loc[self.df[label_name] == 'NeedManualLabel', label_name] = 0
                else:
                    # Original behavior - filter out NeedManualLabel rows when not in prediction mode
                    original_count = len(self.df)
                    self.df = self.df[self.df[label_name] != 'NeedManualLabel']
                    rows_filtered = original_count - len(self.df)
                    print(f"[INFO] Filtered out {rows_filtered} rows with 'NeedManualLabel' ({(rows_filtered/original_count)*100:.2f}% of data)")
    
        # Make sure we have data left after filtering
        if len(self.df) == 0:
            raise ValueError("[ERROR] No data remains after filtering invalid labels. Use prediction_mode=True to keep unlabeled data.")
    
        # Handle label encoding if needed (convert string labels to numeric)
        if self.df[label_name].dtype == object:
            print(f"[INFO] Converting string labels to numeric categories...")
            # Option 1: Use label encoding
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            self.df[label_name] = encoder.fit_transform(self.df[label_name])
        
            # Print mapping for reference
            mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
            print(f"[INFO] Label mapping: {mapping}")
    
        # Try to convert labels to numeric
        try:
            self.df[label_name] = pd.to_numeric(self.df[label_name], errors='coerce')
            # Check for NaN after conversion
            nan_count = self.df[label_name].isna().sum()
            if nan_count > 0:
                print(f"[WARNING] Conversion to numeric created {nan_count} NaN values in label column")
                # Drop rows with NaN labels
                self.df = self.df.dropna(subset=[label_name])
                print(f"[INFO] Dropped {nan_count} rows with NaN labels, {len(self.df)} rows remaining")
        except Exception as e:
            print(f"[ERROR] Failed to convert labels to numeric: {str(e)}")
            raise
    
        # Extract labels and features
        self.y = self.df[label_name].values
        print(f"[DEBUG] Label (y) shape: {self.y.shape}, dtype: {self.y.dtype}")
    
        # Extract features (everything except the label)
        feature_cols = [col for col in self.df.columns if col != label_name]
        self.x = self.df[feature_cols].values
        # First attempt to convert all feature columns to numeric
        for col in feature_cols:
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            except Exception as e:
                print(f"[WARNING] Could not convert column {col} to numeric: {str(e)}")

        # Then extract the features
        self.x = self.df[feature_cols].values

        # Debug info
        print(f"[DEBUG] Column data types after conversion:")
        for col in feature_cols[:5]:  # Print first 5 columns
            print(f"  {col}: {self.df[col].dtype}")

        # Force conversion to float32 before checking for NaN/inf
        try:
            self.x = self.x.astype(np.float32)
            print(f"[DEBUG] Successfully converted features to float32")
        except Exception as e:
            print(f"[ERROR] Could not convert features to float32: {str(e)}")
            # Identify problematic columns
            for i, col in enumerate(feature_cols):
                try:
                    test = self.df[col].values.astype(np.float32)
                except:
                    print(f"  Problem column: {col}, sample value: {self.df[col].iloc[0]}")
        print(f"[DEBUG] Features (X) shape: {self.x.shape}, dtype: {self.x.dtype}")
    
        # Check for invalid values in features
        nan_mask = np.isnan(self.x)
        inf_mask = np.isinf(self.x)
    
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            print(f"[WARNING] Features contain {nan_count} NaN values")
        
            # Identify columns with NaN
            col_nan_counts = np.sum(nan_mask, axis=0)
            problem_cols = [(feature_cols[i], count) for i, count in enumerate(col_nan_counts) if count > 0]
            print(f"[DEBUG] Columns with NaN: {problem_cols}")
        
            # Option: Impute missing values or drop rows
            print("[INFO] Dropping rows with NaN in features")
            valid_rows = ~np.any(nan_mask, axis=1)
            self.x = self.x[valid_rows]
            self.y = self.y[valid_rows]
            print(f"[DEBUG] After dropping NaN rows: X shape {self.x.shape}, y shape {self.y.shape}")
    
        if np.any(inf_mask):
            inf_count = np.sum(inf_mask)
            print(f"[WARNING] Features contain {inf_count} infinite values")
        
            # Identify columns with inf
            col_inf_counts = np.sum(inf_mask, axis=0)
            problem_cols = [(feature_cols[i], count) for i, count in enumerate(col_inf_counts) if count > 0]
            print(f"[DEBUG] Columns with infinite values: {problem_cols}")
        
            # Replace inf with NaN and drop those rows
            print("[INFO] Dropping rows with infinite values in features")
            valid_rows = ~np.any(inf_mask, axis=1)
            self.x = self.x[valid_rows]
            self.y = self.y[valid_rows]
            print(f"[DEBUG] After dropping inf rows: X shape {self.x.shape}, y shape {self.y.shape}")
    
        # Convert to float32 for compatibility with ML libraries
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)
    
        return self.x, self.y
    
    def normalize(self):
        """
        Normalize feature data using MinMaxScaler.
        """
        if self.x is None:
            print("[ERROR] Features (X) are not available for normalization. Call split_x_y first.")
            return None
            
        # Check if X is empty
        if self.x.size == 0:
            print("[ERROR] Features array is empty. Cannot normalize.")
            return None
        
        print(f"[DEBUG] Normalizing features with shape {self.x.shape}")
        scaler = MinMaxScaler()
        
        try:
            self.x = scaler.fit_transform(self.x)
            print("[INFO] Features normalized successfully")
            
            # Debug info on normalized data
            print(f"[DEBUG] Normalized min values: {np.min(self.x, axis=0)[:5]}...")
            print(f"[DEBUG] Normalized max values: {np.max(self.x, axis=0)[:5]}...")
            return scaler  # Return the scaler for future use
        except Exception as e:
            print(f"[ERROR] Normalization failed: {str(e)}")
            return None
    
    def save_processed_data(self, output_file):
        """
        Save the preprocessed data to a CSV file.
        :param output_file: The path to save the processed data.
        """
        if self.x is None or self.y is None:
            print("[ERROR] No processed data available to save.")
            return False
            
        # Check if X and y have elements
        if self.x.size == 0 or self.y.size == 0:
            print("[ERROR] Features or labels array is empty. Cannot save processed data.")
            return False
        
        try:
            # Create a DataFrame with features and label
            feature_names = [f'feature_{i}' for i in range(self.x.shape[1])]
            df_processed = pd.DataFrame(self.x, columns=feature_names)
            df_processed['label'] = self.y
            
            # Save to CSV
            df_processed.to_csv(output_file, index=False)
            print(f"[INFO] Saved processed data to {output_file} ({len(df_processed)} rows, {len(df_processed.columns)} columns)")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save processed data: {str(e)}")
            return False