# dashboard/traffic_processor.py

import json
import asyncio
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import TrafficLog
import joblib
import numpy as np
import os

# Path to your trained ML model (adjust based on your project structure)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml_models', 'ids_model.pkl')

# Try to load the ML model if it exists
try:
    ml_model = joblib.load(MODEL_PATH)
    model_loaded = True
    print("ML model loaded successfully!")
except (FileNotFoundError, Exception) as e:
    model_loaded = False
    print(f"Could not load ML model: {e}")

def prepare_features(traffic_log):
    """Extract and scale features from traffic log for prediction"""
    # Adjust these features according to what your model was trained on
    features = [
        traffic_log.flow_duration or 0,
        traffic_log.tot_fwd_pkts or 0,
        traffic_log.tot_bwd_pkts or 0,
        traffic_log.fwd_pkts_per_sec or 0,
        traffic_log.bwd_pkts_per_sec or 0,
        traffic_log.flow_byts_per_sec or 0,
        traffic_log.flow_pkts_per_sec or 0,
        traffic_log.flow_iat_mean or 0,
        traffic_log.flow_iat_std or 0,
        traffic_log.totlen_fwd_pkts or 0,
        traffic_log.totlen_bwd_pkts or 0,
        traffic_log.fwd_pkt_len_max or 0,
        traffic_log.fwd_pkt_len_min or 0,
        traffic_log.fwd_pkt_len_mean or 0,
        traffic_log.fwd_pkt_len_std or 0,
        traffic_log.bwd_pkt_len_max or 0,
        traffic_log.bwd_pkt_len_min or 0,
        traffic_log.bwd_pkt_len_mean or 0,
        traffic_log.bwd_pkt_len_std or 0,
        traffic_log.flow_iat_max or 0,
        traffic_log.flow_iat_min or 0,
        traffic_log.fwd_iat_tot or 0,
        traffic_log.fwd_iat_mean or 0,
        traffic_log.fwd_iat_std or 0,
        traffic_log.fwd_iat_max or 0,
        traffic_log.fwd_iat_min or 0,
    ]
    
    return np.array([features])

def classify_traffic(traffic_log):
    """Use ML model to classify traffic as normal or attack"""
    if not model_loaded:
        return "unknown"  # Can't classify without model
    
    try:
        # Prepare features for the model
        features = prepare_features(traffic_log)
        
        # Make prediction
        prediction = ml_model.predict(features)[0]
        
        # Return the predicted label
        # Adjust this based on your model's output format
        if prediction == 1:
            return "Suspicious Activity"
        else:
            return None  # Normal traffic
            
    except Exception as e:
        print(f"Error during traffic classification: {e}")
        return "unknown"

@receiver(post_save, sender=TrafficLog)
def process_new_traffic_log(sender, instance, created, **kwargs):
    """Process newly created traffic logs"""
    if not created:
        return  # Skip updates to existing logs
    
    # If no label provided, try to classify with ML model
    if not instance.label and model_loaded:
        label = classify_traffic(instance)
        
        # Update the label if classified as suspicious
        if label and label != "unknown":
            TrafficLog.objects.filter(id=instance.id).update(label=label)
            instance.label = label
    
    # Prepare data for WebSocket broadcast
    log_data = {
        'id': instance.id,
        'timestamp': instance.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'src_ip': instance.src_ip,
        'dst_ip': instance.dst_ip,
        'protocol': instance.protocol,
        'src_port': instance.src_port,
        'dst_port': instance.dst_port
    }
    
    # If this is an alert (has a label), prepare alert broadcast
    if instance.label:
        # Determine severity based on label
        severity = 'low'
        if any(kw in instance.label.lower() for kw in ['attack', 'injection', 'xss', 'ddos']):
            severity = 'high'
        elif any(kw in instance.label.lower() for kw in ['scan', 'probe', 'brute']):
            severity = 'medium'
        
        alert_data = {**log_data, 'label': instance.label, 'severity': severity}
        
        # Send alert update via WebSocket
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            "dashboard_updates",
            {
                'type': 'alert_update',
                'data': alert_data
            }
        )
    
    # Send general traffic update via WebSocket
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        "dashboard_updates",
        {
            'type': 'traffic_update',
            'data': log_data
        }
    )