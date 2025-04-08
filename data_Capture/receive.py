import os
import time
import pika
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import psycopg2
import psycopg2.extras
from processing_analysis.netflow_converter import PcapToNetFlow
from processing_analysis.classifier import NeuralNetworkClassifier
from processing_analysis.preprocessor import PreProcessCICFlow

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
        print(f"Processing file: {filename}")
        start_time = time.time()

        # Convert pcap to NetFlow CSV
        converter = PcapToNetFlow(filename)
        csv_file = converter.convert()
        print(f"[INFO] Converted pcap to CSV: {csv_file}")

        # Preprocess the CSV file using PreProcessCICFlow
        from processing_analysis.preprocessor import PreProcessCICFlow
        preprocessor = PreProcessCICFlow(csv_file)
        
        # Set columns to drop
        drop_columns = ['Flow ID', 'Timestamp', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol']
        preprocessor.set_drop_columns(drop_columns)
        preprocessor.drop_unused_columns()

        # Preprocess the data
        preprocessor.preprocess()
        x, y = preprocessor.split_x_y('Label')

        # Normalize features
        preprocessor.normalize()

        # Classify traffic using the loaded model
        predictions = nnc.predict(x)
        print(f"Predictions: {predictions}")

        # Load the CSV into a DataFrame and store results
        df = pd.read_csv(csv_file)
        df['Label'] = predictions
        df['Label'] = df['Label'].map({0: "Normal", 1: "Threat"})

        # Save results to the database
        save_to_database(df)

        # Count threats and print summary
        total_threats = (df['Label'] == "Threat").sum()
        if total_threats > 0:
            print(f"[*] ALERT: {total_threats} threats detected!")

        # Cleanup generated CSV file
        os.remove(csv_file)

        print(f"Processing completed in {time.time() - start_time:.2f} seconds.")

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

def save_to_database(df):
    """
    Save the classified DataFrame to the PostgreSQL database.
    """
    try:
        cur = conn.cursor()
        columns = [col.replace(" ", "_").replace("/", "_per_").lower() for col in df.columns]
        table = "dash_trafficlog"

        # Prepare SQL statement
        columns_str = ", ".join(columns)
        values_str = ", ".join(["%s"] * len(columns))
        insert_stmt = f"INSERT INTO {table} ({columns_str}) VALUES ({values_str})"

        # Execute batch insert
        psycopg2.extras.execute_batch(cur, insert_stmt, df.values)
        conn.commit()
        cur.close()
        print("Data successfully saved to the database.")

    except Exception as e:
        print(f"Error saving to database: {e}")

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
