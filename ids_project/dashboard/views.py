# views.py
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import TrafficLog
from .serializers import TrafficLogSerializer
from .threat_engine import alert_severity_from_stored_fields
from django.db.models import Count
from django.db import models
from django.utils import timezone
import datetime

def index(request):
    return render(request, 'dashboard/index.html')

class TrafficLogViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = TrafficLog.objects.all().order_by('-timestamp')
    serializer_class = TrafficLogSerializer

@api_view(['GET'])
def traffic_over_time(request):
    """Get traffic counts per hour for the last 24 hours"""
    end_time = timezone.now()
    start_time = end_time - datetime.timedelta(days=1)
    
    # Get normal and suspicious traffic counts by hour
    traffic_data = []
    
    for hour in range(24):
        hour_start = start_time + datetime.timedelta(hours=hour)
        hour_end = hour_start + datetime.timedelta(hours=1)
        
        normal_count = TrafficLog.objects.filter(
            timestamp__gte=hour_start,
            timestamp__lt=hour_end,
        ).filter(
            models.Q(label__isnull=True) | models.Q(label='') | models.Q(label='Normal')
        ).count()
        
        suspicious_count = TrafficLog.objects.filter(
            timestamp__gte=hour_start,
            timestamp__lt=hour_end,
        ).exclude(
            models.Q(label__isnull=True) | models.Q(label='') | models.Q(label='Normal')
        ).count()
        
        traffic_data.append({
            'hour': hour,
            'normal': normal_count,
            'suspicious': suspicious_count
        })
    
    return Response(traffic_data)

@api_view(['GET'])
def protocol_distribution(request):
    """Get distribution of protocols in the traffic data"""
    protocols = TrafficLog.objects.values('protocol').annotate(
        value=Count('protocol')
    ).order_by('-value')
    
    # Format for pie chart
    protocol_data = [{'name': p['protocol'] or 'Unknown', 'value': p['value']} for p in protocols]
    
    return Response(protocol_data)

@api_view(['GET'])
def top_sources(request):
    """Get top source IPs by traffic count"""
    top_ips = TrafficLog.objects.values('src_ip').annotate(
        count=Count('src_ip')
    ).order_by('-count')[:10]
    
    return Response(top_ips)

@api_view(['GET'])
def recent_alerts(request):
    """Get recent traffic logs with alert labels"""
    alerts = TrafficLog.objects.exclude(
        models.Q(label__isnull=True) | models.Q(label='') | models.Q(label='Normal')
    ).order_by('-timestamp')[:20]
    
    # Add severity based on label (could be enhanced with ML classification)
    alert_data = []
    for alert in alerts:
        severity = alert_severity_from_stored_fields(
            alert.label, alert.threat_type, alert.threat_family
        )
        alert_data.append({
            'id': alert.id,
            'timestamp': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip': alert.src_ip,
            'dst_ip': alert.dst_ip,
            'protocol': alert.protocol,
            'label': alert.label,
            'threat_type': alert.threat_type or '',
            'threat_family': alert.threat_family or '',
            'threat_detail': alert.threat_detail or '',
            'detection_source': alert.detection_source or '',
            'confidence': float(alert.confidence) if alert.confidence is not None else None,
            'severity': severity,
        })
    
    return Response(alert_data)

@api_view(['GET'])
def dashboard_stats(request):
    """Get summary statistics for the dashboard"""
    end_time = timezone.now()
    start_time = end_time - datetime.timedelta(days=1)
    
    total_traffic = TrafficLog.objects.filter(timestamp__gte=start_time).count()
    
    alerts_count = TrafficLog.objects.filter(
        timestamp__gte=start_time,
    ).exclude(
        models.Q(label__isnull=True) | models.Q(label='') | models.Q(label='Normal')
    ).count()
    
    active_hosts = TrafficLog.objects.filter(
        timestamp__gte=start_time
    ).values('src_ip').distinct().count()
    
    # Calculate total data transferred (sum of forward and backward packet lengths)
    data_transferred = TrafficLog.objects.filter(timestamp__gte=start_time).aggregate(
        total_bytes=models.Sum(models.F('totlen_fwd_pkts') + models.F('totlen_bwd_pkts'))
    )['total_bytes'] or 0
    
    # Convert to GB
    data_transferred_gb = data_transferred / (1024 * 1024 * 1024)
    
    return Response({
        'total_traffic': total_traffic,
        'alerts_count': alerts_count,
        'active_hosts': active_hosts,
        'data_transferred': round(data_transferred_gb, 2)  # GB to 2 decimal places
    })