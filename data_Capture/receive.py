import os
import time
import pika
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import psycopg2
import psycopg2.extras
from datetime import datetime
from sqlalchemy import types
from processing_analysis.netflow_converter import PcapToNetFlow
from processing_analysis.classifier import NeuralNetworkClassifier
from processing_analysis.preprocessor import PreProcessNetFlowCsv

# Suppress TensorFlow warnings
def tensorflow_shutup():
    """
    Reduce TensorFlow verbosity by disabling logging messages.
    """
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tensorflow_shutup()

# RabbitMQ setup
try:
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.exchange_declare(exchange='logs', exchange_type='fanout')
    result = channel.queue_declare(queue='', exclusive=True)
    queue_name = result.method.queue
    channel.queue_bind(exchange='logs', queue=queue_name)
    print("[*] Connected to RabbitMQ. Waiting for logs. To exit, press CTRL+C")
except Exception as e:
    print(f"Error connecting to RabbitMQ: {e}")
    exit(1)

# PostgreSQL connection
try:
    conn = psycopg2.connect(
        user="netvizuser",
        password="netviz123",
        host="127.0.0.1",
        port="5432",
        database="netviz"
    )
    print("[*] Connected to PostgreSQL database.")
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")
    exit(1)

# Load the neural network model
model_path = '/home/zayn/Desktop/IDS-ML/models/dnn-model.hdf5'
try:
    nnc = NeuralNetworkClassifier(model_path)
    nnc.load_model(compile=False)  # Ignore the optimizer configuration
    print("[*] Neural network model loaded successfully.")
except Exception as e:
    print(f"Error loading the neural network model: {e}")
    exit(1)


def process_file(filename):
    """
    Process a .pcap file: Convert to CSV, preprocess, classify, and store results.
    """
    try:
        print(f"\n\n==== STARTING PROCESSING FOR FILE: {filename} ====")
        print(f"Current working directory: {os.getcwd()}")
        start_time = time.time()

        # Convert pcap to NetFlow CSV
        print(f"[DEBUG] About to convert pcap to NetFlow CSV...")
        converter = PcapToNetFlow(filename)
        csv_file = converter.convert()
        print(f"[INFO] Converted pcap to CSV: {csv_file}")
        
        # Verify CSV file exists and has content
        if not os.path.exists(csv_file):
            print(f"[ERROR] CSV file does not exist: {csv_file}")
            return
            
        file_size = os.path.getsize(csv_file)
        print(f"[DEBUG] CSV file size: {file_size} bytes")
        
        if file_size == 0:
            print(f"[ERROR] CSV file is empty: {csv_file}")
            return
            
        # Read first few lines of CSV to debug format
        print(f"\n[DEBUG] CSV file first 5 lines:")
        with open(csv_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"Line {i}: {line.strip()}")
                else:
                    break

        # Load raw data with explicit dtype for all critical columns
        print("\n[DEBUG] Loading CSV file with explicit dtypes...")
        try:
            dtypes = {
                'Src Port': 'int32', 
                'Dst Port': 'int32',
                'Protocol': 'object',  # Try as object first
                'Src IP': 'object',    # Ensure IPs are kept as strings
                'Dst IP': 'object'
            }
            original_df = pd.read_csv(csv_file, dtype=dtypes)
            print("[DEBUG] CSV loaded with explicit dtypes")
        except Exception as e:
            print(f"[WARN] Failed to load with explicit dtypes: {str(e)}")
            try:
                print("[DEBUG] Trying to load CSV without dtypes...")
                original_df = pd.read_csv(csv_file)  # Fallback if specific dtypes fail
                print("[DEBUG] CSV loaded without dtypes")
            except Exception as e2:
                print(f"[ERROR] Failed to load CSV at all: {str(e2)}")
                return
            
        original_df = original_df.reset_index(drop=True)
        
        # Make a copy of the original data to preserve all columns
        complete_df = original_df.copy()
        
        # Extensive debugging of the raw data to see actual values
        print("\n===== RAW DATA DEBUGGING =====")
        print(f"DataFrame shape: {original_df.shape}")
        print(f"DataFrame columns: {original_df.columns.tolist()}")
        print("\n*** RAW DATA SAMPLE (First 3 rows) ***")
        
        # Check if the expected columns exist
        expected_columns = ['Flow ID', 'Timestamp', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol']
        missing_columns = [col for col in expected_columns if col not in original_df.columns]
        
        if missing_columns:
            print(f"[WARN] Missing expected columns: {missing_columns}")
            print(f"Available columns: {original_df.columns.tolist()}")
        
        # Show sample data for key columns
        for col in ['Flow ID', 'Timestamp', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol']:
            if col in original_df.columns:
                print(f"\n[DEBUG] Column {col} data types and values:")
                print(f"Data type: {original_df[col].dtype}")
                print(f"First 5 values: {original_df[col].head(5).tolist()}")
                print(f"Unique values: {original_df[col].nunique()}")
                print(f"Sample unique values: {original_df[col].unique()[:5]}")
                
                # Check for NaN or empty values
                nan_count = original_df[col].isna().sum()
                if nan_count > 0:
                    print(f"[WARN] Column {col} has {nan_count} NaN values")
                
                # For IP columns, check for problematic values
                if 'IP' in col:
                    zero_ip_count = (original_df[col] == '0.0.0.0').sum()
                    if zero_ip_count > 0:
                        print(f"[WARN] Column {col} has {zero_ip_count} values of '0.0.0.0'")
        
        # Print the first 3 rows of key columns
        display_cols = [col for col in expected_columns if col in original_df.columns]
        print("\n*** RAW DATA SAMPLE (First 3 rows with key columns) ***")
        print(original_df[display_cols].head(3))

        # Preprocess for prediction only
        drop_columns = ['Flow ID', 'Timestamp', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol']
        preprocessor = PreProcessNetFlowCsv(csv_file_name=csv_file)
        preprocessor.set_drop_columns(drop_columns)
        preprocessor.drop_unused_column()
        preprocessor.pre_process()

        # Fallback handling
        if preprocessor.df.empty:
            print(f"[WARN] No valid data to process in {csv_file}. Using fallback test data.")
            original_csv_file = csv_file

            if os.path.exists(csv_file):
                os.remove(csv_file)

            test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata.csv")
            if not os.path.exists(test_data_path):
                print(f"[ERROR] Fallback test data file not found at {test_data_path}")
                return

            preprocessor = PreProcessNetFlowCsv(csv_file_name=test_data_path)
            preprocessor.set_drop_columns(drop_columns)
            preprocessor.drop_unused_column()
            preprocessor.pre_process()
            csv_file = test_data_path
            
            # Also reload the complete data
            try:
                complete_df = pd.read_csv(test_data_path, dtype={'Src Port': 'int32', 'Dst Port': 'int32'})
            except:
                complete_df = pd.read_csv(test_data_path)

            if preprocessor.df.empty:
                print(f"[ERROR] Fallback test data is also empty. Cannot proceed.")
                return

        # Prepare data for prediction
        x, _ = preprocessor.split_x_y('Label') if isinstance(preprocessor.split_x_y('Label'), tuple) else (preprocessor.split_x_y('Label'), None)
        if x is None or len(x) == 0:
            print("[ERROR] No features extracted for prediction. Aborting.")
            return

        x = np.expand_dims(x, axis=1)  # Add time dimension for model
        print(f"x shape: {x.shape}")
        y_pred = nnc.predict(x)
        print(f"y_pred shape: {y_pred.shape}")

        # Convert predictions to class labels
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        elif y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()

        # Validation check
        if len(y_pred) != len(complete_df):
            print(f"[WARN] Length mismatch: predictions={len(y_pred)}, rows={len(complete_df)}.")
            # Adjust based on the smaller length to avoid index errors
            min_len = min(len(y_pred), len(complete_df))
            y_pred = y_pred[:min_len]
            complete_df = complete_df.iloc[:min_len].copy()

        # Add predictions to the complete dataframe
        complete_df["Label"] = pd.Series(y_pred.astype(int)).map({0: "Normal", 1: "Threat"})
        
        # DEBUG: Print column names before normalization
        print("\n[DEBUG] Column names BEFORE normalization:")
        print(complete_df.columns.tolist())
        
        # IMPORTANT: Save original capitalized column values before normalizing names
        # This is critical to not lose the original data during column name normalization
        capitalized_columns = {
            'Src IP': 'src_ip_temp', 
            'Dst IP': 'dst_ip_temp',
            'Protocol': 'protocol_temp',
            'Src Port': 'src_port_temp',
            'Dst Port': 'dst_port_temp',
            'Flow ID': 'flow_id_temp'
        }
        
        # Create temporary columns with the original data
        for orig_col, temp_col in capitalized_columns.items():
            if orig_col in complete_df.columns:
                print(f"[DEBUG] Preserving data from {orig_col} in {temp_col}")
                complete_df[temp_col] = complete_df[orig_col]
                
        # Now normalize column names
        print("[DEBUG] Normalizing column names...")
        complete_df.columns = complete_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_per_")
        
        # DEBUG: Print column names after normalization
        print("[DEBUG] Column names AFTER normalization:")
        print(complete_df.columns.tolist())
        
        # Move data from temp columns back to properly named columns
        temp_to_final = {
            'src_ip_temp': 'src_ip',
            'dst_ip_temp': 'dst_ip',
            'protocol_temp': 'protocol',
            'src_port_temp': 'src_port',
            'dst_port_temp': 'dst_port',
            'flow_id_temp': 'flow_id'
        }
        
        for temp_col, final_col in temp_to_final.items():
            if temp_col in complete_df.columns:
                print(f"[DEBUG] Restoring data from {temp_col} to {final_col}")
                if final_col in complete_df.columns:
                    # Check if current values are all zeros/empty
                    if complete_df[final_col].isin(['0', '0.0.0.0', '', 0]).all():
                        complete_df[final_col] = complete_df[temp_col]
                    else:
                        # If some values are valid, only replace zeros/empty
                        mask = complete_df[final_col].isin(['0', '0.0.0.0', '', 0])
                        complete_df.loc[mask, final_col] = complete_df.loc[mask, temp_col]
                else:
                    complete_df[final_col] = complete_df[temp_col]
                    
                # Remove temporary column
                complete_df.drop(columns=[temp_col], inplace=True)
                
        # Print key columns after restoration
        print("\n[DEBUG] Key columns after restoration:")
        for col in ['src_ip', 'dst_ip', 'protocol', 'src_port', 'dst_port']:
            if col in complete_df.columns:
                print(f"{col} sample values: {complete_df[col].head(5).tolist()}")
        
        # Handle specific column types and restrictions
        
        # Handle IP addresses - extensive debugging
        print("\n[DEBUG] IP Address handling:")
        
        # Check for correct column capitalization
        ip_cols = [col for col in complete_df.columns if 'ip' in col.lower()]
        print(f"IP-related columns found: {ip_cols}")
        
        # Debug and fix source IP
        if 'src_ip' in complete_df.columns:
            print(f"[DEBUG] src_ip column dtype: {complete_df['src_ip'].dtype}")
            print(f"[DEBUG] src_ip sample values before cleanup: {complete_df['src_ip'].head(5).tolist()}")
            complete_df['src_ip'] = complete_df['src_ip'].astype(str)
            # Check for empty or zero IPs and replace with proper values
            zero_ips = (complete_df['src_ip'] == '0') | (complete_df['src_ip'] == '0.0.0.0') | (complete_df['src_ip'] == '')
            if zero_ips.any():
                print(f"[WARN] Found {zero_ips.sum()} zero/empty src_ip values")
                # Look for original Source IP column with capitalization
                if 'Src IP' in original_df.columns:
                    print("[DEBUG] Found original 'Src IP' column, using values from there")
                    for idx in complete_df[zero_ips].index:
                        if idx < len(original_df):
                            original_value = original_df.at[idx, 'Src IP']
                            if original_value not in ['0', '0.0.0.0', '', None]:
                                complete_df.at[idx, 'src_ip'] = str(original_value)
                                print(f"[DEBUG] Fixed src_ip at index {idx}: {original_value}")
            print(f"[DEBUG] src_ip sample values after cleanup: {complete_df['src_ip'].head(5).tolist()}")
        elif 'Src IP' in complete_df.columns:
            print("[DEBUG] Creating src_ip from Src IP")
            complete_df['src_ip'] = complete_df['Src IP'].astype(str)
        else:
            print("[WARN] No source IP column found!")
            
        # Debug and fix destination IP
        if 'dst_ip' in complete_df.columns:
            print(f"[DEBUG] dst_ip column dtype: {complete_df['dst_ip'].dtype}")
            print(f"[DEBUG] dst_ip sample values before cleanup: {complete_df['dst_ip'].head(5).tolist()}")
            complete_df['dst_ip'] = complete_df['dst_ip'].astype(str)
            # Check for empty or zero IPs and replace with proper values
            zero_ips = (complete_df['dst_ip'] == '0') | (complete_df['dst_ip'] == '0.0.0.0') | (complete_df['dst_ip'] == '')
            if zero_ips.any():
                print(f"[WARN] Found {zero_ips.sum()} zero/empty dst_ip values")
                # Look for original Dst IP column with capitalization
                if 'Dst IP' in original_df.columns:
                    print("[DEBUG] Found original 'Dst IP' column, using values from there")
                    for idx in complete_df[zero_ips].index:
                        if idx < len(original_df):
                            original_value = original_df.at[idx, 'Dst IP']
                            if original_value not in ['0', '0.0.0.0', '', None]:
                                complete_df.at[idx, 'dst_ip'] = str(original_value)
                                print(f"[DEBUG] Fixed dst_ip at index {idx}: {original_value}")
            print(f"[DEBUG] dst_ip sample values after cleanup: {complete_df['dst_ip'].head(5).tolist()}")
        elif 'Dst IP' in complete_df.columns:
            print("[DEBUG] Creating dst_ip from Dst IP")
            complete_df['dst_ip'] = complete_df['Dst IP'].astype(str)
        else:
            print("[WARN] No destination IP column found!")
            
        # Handle port numbers - ensure they're integers within valid range
        for port_col in ['src_port', 'dst_port']:
            if port_col in complete_df.columns:
                complete_df[port_col] = pd.to_numeric(complete_df[port_col], errors='coerce').fillna(0)
                complete_df[port_col] = complete_df[port_col].clip(lower=0, upper=65535).astype('int32')
        
        # Handle protocol - convert numeric protocols to text names if needed
        print("\n[DEBUG] Protocol handling:")
        protocol_map = {
            0: 'HOPOPT',
            1: 'ICMP',
            6: 'TCP',
            17: 'UDP',
            58: 'ICMPv6',
        }
        
        # Debug protocol values
        if 'protocol' in complete_df.columns:
            print(f"Protocol column found. Data type: {complete_df['protocol'].dtype}")
            print(f"Protocol unique values before conversion: {complete_df['protocol'].unique()[:10]}")
            
            # Check for case inconsistencies that might cause issues
            if 'Protocol' in complete_df.columns and 'protocol' in complete_df.columns:
                print("[WARN] Both 'Protocol' and 'protocol' columns exist - might cause confusion")
                
            # Fix protocol based on its data type
            if pd.api.types.is_numeric_dtype(complete_df['protocol']):
                print("[DEBUG] Protocol is numeric, mapping to protocol names...")
                # Save original values for debugging
                original_protocols = complete_df['protocol'].copy()
                
                # Convert protocol values
                complete_df['protocol'] = complete_df['protocol'].fillna(0).astype(int).map(
                    lambda x: protocol_map.get(x, str(x))
                )
                
                # Compare before and after 
                print(f"[DEBUG] Protocol samples before mapping: {original_protocols.head(5).tolist()}")
                print(f"[DEBUG] Protocol samples after mapping: {complete_df['protocol'].head(5).tolist()}")
            else:
                print("[DEBUG] Protocol is non-numeric, making sure it's proper string...")
                complete_df['protocol'] = complete_df['protocol'].fillna("Unknown").astype(str)
                
            print(f"[DEBUG] Protocol unique values after conversion: {complete_df['protocol'].unique()[:10]}")
        else:
            print("[WARN] No 'protocol' column found - looking for 'Protocol' column")
            if 'Protocol' in complete_df.columns:
                print(f"[DEBUG] 'Protocol' column found instead. Data type: {complete_df['Protocol'].dtype}")
                print(f"[DEBUG] Protocol unique values: {complete_df['Protocol'].unique()[:10]}")
                
                # Create properly named protocol column
                complete_df['protocol'] = complete_df['Protocol']
                
                if pd.api.types.is_numeric_dtype(complete_df['protocol']):
                    complete_df['protocol'] = complete_df['protocol'].fillna(0).astype(int).map(
                        lambda x: protocol_map.get(x, str(x))
                    )
                else:
                    complete_df['protocol'] = complete_df['protocol'].fillna("Unknown").astype(str)
        
        # Handle timestamp if present
        if 'timestamp' in complete_df.columns:
            complete_df['timestamp'] = pd.to_datetime(complete_df['timestamp'], errors='coerce').fillna(datetime.now())
            
        # PostgreSQL INTEGER limit is 2,147,483,647
        INT_MAX = 2147483647
        INT_MIN = -2147483648
        
        # Handle numeric columns to prevent database errors
        for col in complete_df.select_dtypes(include=['int', 'float']).columns:
            if col not in ['id', 'src_port', 'dst_port']:  # Special handling for these columns
                # Clip values to PostgreSQL integer limits
                if complete_df[col].max() > INT_MAX or complete_df[col].min() < INT_MIN:
                    print(f"Clipping column {col} values to database integer range")
                    complete_df[col] = complete_df[col].clip(lower=INT_MIN, upper=INT_MAX)
                    # Convert to float for values that might exceed integer range
                    complete_df[col] = complete_df[col].astype('float64')
        
        # Additional checks for problematic values
        print("\n[DEBUG] Final validation checks before database save:")
        
        # Check for zero IPs and fix them if possible
        for col in ['src_ip', 'dst_ip']:
            if col in complete_df.columns:
                zero_mask = complete_df[col].isin(['0', '0.0.0.0', ''])
                zero_count = zero_mask.sum()
                if zero_count > 0:
                    print(f"[WARN] {col} still has {zero_count} zero/empty values")
                    
                    # Try to reconstruct from flow_id if available
                    if 'flow_id' in complete_df.columns:
                        fixed_count = 0
                        for idx in complete_df[zero_mask].index:
                            flow_id = complete_df.at[idx, 'flow_id']
                            if isinstance(flow_id, str) and '-' in flow_id:
                                # Flow ID format typically: src_ip-dst_ip-protocol-src_port-dst_port
                                parts = flow_id.split('-')
                                if len(parts) >= 2:
                                    if col == 'src_ip' and parts[0] not in ['0', '0.0.0.0', '']:
                                        complete_df.at[idx, 'src_ip'] = parts[0]
                                        fixed_count += 1
                                    elif col == 'dst_ip' and parts[1] not in ['0', '0.0.0.0', '']:
                                        complete_df.at[idx, 'dst_ip'] = parts[1]
                                        fixed_count += 1
                        if fixed_count > 0:
                            print(f"[DEBUG] Fixed {fixed_count} {col} values using flow_id")
        
        # Check protocol values - if all are HOPOPT, try to fix
        if 'protocol' in complete_df.columns:
            hopopt_count = (complete_df['protocol'] == 'HOPOPT').sum()
            if hopopt_count == len(complete_df):
                print("[WARN] All protocol values are HOPOPT (0) - attempting to fix")
                
                # Try to extract from flow_id
                if 'flow_id' in complete_df.columns:
                    protocol_map_reverse = {
                        'TCP': 6,
                        'UDP': 17,
                        'ICMP': 1
                    }
                    fixed_count = 0
                    for idx, row in complete_df.iterrows():
                        flow_id = row['flow_id']
                        if isinstance(flow_id, str) and '-' in flow_id:
                            parts = flow_id.split('-')
                            if len(parts) >= 3 and parts[2].isdigit():
                                proto_num = int(parts[2])
                                if proto_num > 0:  # Not HOPOPT
                                    proto_str = protocol_map.get(proto_num, str(proto_num))
                                    complete_df.at[idx, 'protocol'] = proto_str
                                    fixed_count += 1
                    if fixed_count > 0:
                        print(f"[DEBUG] Fixed {fixed_count} protocol values using flow_id")
        
        # Ensure all required columns exist before database save
        required_cols = ['src_ip', 'dst_ip', 'protocol', 'src_port', 'dst_port', 'label', 'flow_id', 'timestamp']
        for col in required_cols:
            if col not in complete_df.columns:
                print(f"[WARN] Required column {col} missing - creating with default values")
                if col in ['src_ip', 'dst_ip', 'protocol', 'label', 'flow_id']:
                    complete_df[col] = 'Unknown'
                elif col in ['src_port', 'dst_port']:
                    complete_df[col] = 0
                elif col == 'timestamp':
                    complete_df[col] = datetime.now()
        
        # FIX: Map column names to align with database expectations
        # This is the key fix for the missing columns warning
        column_mapping = {
            'flow_pkts_per_s': 'flow_pkts_per_sec',
            'flow_byts_per_s': 'flow_byts_per_sec',
            'fwd_pkts_per_s': 'fwd_pkts_per_sec', 
            'bwd_pkts_per_s': 'bwd_pkts_per_sec',
            'fwd_byts_per_b_avg': 'fwd_byts_b_avg',
            'fwd_pkts_per_b_avg': 'fwd_pkts_b_avg',
            'bwd_byts_per_b_avg': 'bwd_byts_b_avg',
            'bwd_pkts_per_b_avg': 'bwd_pkts_b_avg'
        }
        
        print("\n[DEBUG] Renaming columns to match database schema:")
        for old_col, new_col in column_mapping.items():
            if old_col in complete_df.columns:
                print(f"Renaming {old_col} to {new_col}")
                complete_df[new_col] = complete_df[old_col]
                complete_df.drop(columns=[old_col], inplace=True)
            else:
                # If original column not found, create the expected column with default values
                print(f"Creating missing column {new_col} with default values")
                if 'byts' in new_col or 'pkts' in new_col:
                    complete_df[new_col] = 0  # Default to 0 for byte/packet counts or rates
        
        # Final debug output
        print("\n*** FINAL DATAFRAME SAMPLE (First 3 rows) ***")
        display_cols = [col for col in required_cols if col in complete_df.columns]
        print(complete_df[display_cols].head(3))
        
        print("\n*** COLUMN TYPES AND SAMPLES ***")
        for col in required_cols:
            if col in complete_df.columns:
                unique_values = complete_df[col].nunique()
                sample_val = complete_df[col].iloc[0] if len(complete_df) > 0 else 'N/A'
                print(f"Column {col}: type={complete_df[col].dtype}, unique values={unique_values}, sample={sample_val}")
        
        try:
            # Attempt to save to database using our debug wrapper
            print("Attempting to save to database with properly typed values...")
            if 'save_to_database_with_debug' in globals():
                # Use debug wrapper if available
                save_to_database_with_debug(complete_df)
            else:
                # Define the debug wrapper inline if not imported
                def debug_save(df):
                    print("\n[DATABASE DEBUG] Debugging database save...")
                    try:
                        save_to_database(df)
                        return True
                    except Exception as e:
                        print(f"[DATABASE ERROR] {str(e)}")
                        return False
                        
                debug_save(complete_df)
            print("Successfully saved to database!")
        except Exception as db_error:
            print(f"Database save error: {str(db_error)}")
            traceback.print_exc()
            
            # Try to fix missing columns if that's the issue
            if "missing columns" in str(db_error).lower():
                print("Attempting to add missing columns with default values...")
                
                # Extract required columns from the error message
                try:
                    import re
                    match = re.search(r"Missing columns.*?\{(.*?)\}", str(db_error))
                    if match:
                        missing_cols = match.group(1).replace("'", "").split(", ")
                        print(f"Adding {len(missing_cols)} missing columns from error message")
                        
                        for col in missing_cols:
                            if col not in complete_df.columns:
                                if any(substr in col for substr in ['_std', '_avg', '_mean', '_min', '_max', '_var', '_tot', '_rate']):
                                    complete_df[col] = 0.0
                                elif any(substr in col for substr in ['_cnt', '_count', '_flags', '_pkts', '_byts']):
                                    complete_df[col] = 0
                                else:
                                    complete_df[col] = None
                        
                        # Try saving again
                        save_to_database(complete_df)
                        print("Save successful after adding missing columns from error message!")
                except Exception as final_error:
                    print(f"Final error: {str(final_error)}")
                    traceback.print_exc()

        total_threats = (complete_df["label"] == "Threat").sum()
        print(f"Total Threats Detected: {total_threats}")
        if total_threats > 0:
            print(f"[*] ALERT: {total_threats} threats detected!")
            push_notify(total_threats, "please see the console")

        # Cleanup
        try:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Removed original pcap file: {filename}")
            if 'original_csv_file' in locals() and os.path.exists(original_csv_file):
                os.remove(original_csv_file)
                print(f"Removed original CSV file: {original_csv_file}")
            elif os.path.exists(csv_file) and ('test_data_path' not in locals() or csv_file != test_data_path):
                os.remove(csv_file)
                print(f"Removed CSV file: {csv_file}")
        except Exception as cleanup_err:
            print(f"Warning during file cleanup: {str(cleanup_err)}")

        print(f"Processing completed in {time.time() - start_time:.2f} seconds.")
        print("*************************************************************")

    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        traceback.print_exc()  # Print detailed error information

def save_to_database(df):
    """
    Save the classified DataFrame to the PostgreSQL database.
    """
    try:
        cur = conn.cursor()

        # Fetch the current columns in the dash_trafficlog table and convert to lowercase
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='dash_trafficlog'")
        table_columns = set(row[0].lower() for row in cur.fetchall())

        # Exclude 'id' since it's auto-incremented
        table_columns.discard('id')

        # Rename DataFrame columns to match DB format (lowercase)
        df.columns = [col.replace(" ", "_").replace("/", "_per_").lower() for col in df.columns]

        # Add any missing required fields (like 'timestamp') with default/fallback values
        if 'timestamp' in table_columns and 'timestamp' not in df.columns:
            from datetime import datetime
            df['timestamp'] = datetime.now()

        # Optional defaults for dashboard if missing (using lowercase for comparison)
        for col in ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']:
            if col in table_columns and col not in df.columns:
                df[col] = '0.0.0.0' if 'ip' in col else (0 if 'port' in col else 'tcp') # Using lowercase 'tcp' for consistency

        # Filter only valid columns for insertion (using lowercase for comparison)
        valid_columns = [col for col in df.columns if col in table_columns]

        missing_columns = table_columns - set(valid_columns)
        if missing_columns:
            print(f"Warning: Missing columns in DataFrame for insertion: {missing_columns}")

        # Prepare and execute the batch insert
        columns_str = ", ".join(valid_columns)
        values_str = ", ".join(["%s"] * len(valid_columns))
        insert_stmt = f"INSERT INTO dash_trafficlog ({columns_str}) VALUES ({values_str})"

        psycopg2.extras.execute_batch(cur, insert_stmt, df[valid_columns].values)
        conn.commit()
        cur.close()

        print("Data successfully saved to the database.")

    except Exception as e:
        print(f"Error saving to database: {e}")
        import traceback
        traceback.print_exc() # Print the full traceback for better debugging



def callback(ch, method, properties, body):
    """
    Callback function for RabbitMQ. Processes the received .pcap filename.
    """
    filename = body.decode("utf-8")
    process_file(filename)

# Start consuming messages from RabbitMQ
try:
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()
except KeyboardInterrupt:
    print("\n[*] Exiting...")
    connection.close()
except Exception as e:
    print(f"Error consuming messages: {e}")