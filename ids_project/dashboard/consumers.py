# consumers.py (for WebSocket real-time updates)
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import TrafficLog
from django.db.models import Count
from django.utils import timezone
import datetime

class DashboardConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add(
            "dashboard_updates",
            self.channel_name
        )
        await self.accept()
        
        # Send initial data
        await self.send_initial_data()
        
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            "dashboard_updates",
            self.channel_name
        )
    
    async def send_initial_data(self):
        # Get initial dashboard data
        traffic_data = await self.get_traffic_data()
        protocol_data = await self.get_protocol_data()
        top_sources = await self.get_top_sources()
        recent_alerts = await self.get_recent_alerts()
        stats = await self.get_dashboard_stats()
        
        # Send to client
        await self.send(text_data=json.dumps({
            'trafficOverTime': traffic_data,
            'protocolDistribution': protocol_data,
            'topSources': top_sources,
            'recentAlerts': recent_alerts,
            'stats': stats
        }))
    
    @database_sync_to_async
    def get_traffic_data(self):
        # Similar to traffic_over_time view
        end_time = timezone.now()
        start_time = end_time - datetime.timedelta(days=1)
        
        traffic_data = []
        for hour in range(24):
            hour_start = start_time + datetime.timedelta(hours=hour)
            hour_end = hour_start + datetime.timedelta(hours=1)
            
            normal_count = TrafficLog.objects.filter(
                timestamp__gte=hour_start,
                timestamp__lt=hour_end,
                label__isnull=True
            ).count()
            
            suspicious_count = TrafficLog.objects.filter(
                timestamp__gte=hour_start,
                timestamp__lt=hour_end,
                label__isnull=False
            ).count()
            
            traffic_data.append({
                'hour': hour,
                'normal': normal_count,
                'suspicious': suspicious_count
            })
        
        return traffic_data
    
    @database_sync_to_async
    def get_protocol_data(self):
        # Similar to protocol_distribution view
        protocols = TrafficLog.objects.values('protocol').annotate(
            value=Count('protocol')
        ).order_by('-value')
        
        return [{'name': p['protocol'] or 'Unknown', 'value': p['value']} for p in protocols]
    
    @database_sync_to_async
    def get_top_sources(self):
        # Similar to top_sources view
        top_ips = TrafficLog.objects.values('src_ip').annotate(
            count=Count('src_ip')
        ).order_by('-count')[:10]
        
        return list(top_ips)
    
    @database_sync_to_async
    def get_recent_alerts(self):
        # Similar to recent_alerts view
        alerts = TrafficLog.objects.exclude(
            label__isnull=True
        ).exclude(
            label=''
        ).order_by('-timestamp')[:20]
        
        alert_data = []
        for alert in alerts:
            severity = 'low'
            if any(kw in alert.label.lower() for kw in ['attack', 'injection', 'xss', 'ddos']):
                severity = 'high'
            elif any(kw in alert.label.lower() for kw in ['scan', 'probe', 'brute']):
                severity = 'medium'
                
            alert_data.append({
                'id': alert.id,
                'timestamp': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'src_ip': alert.src_ip,
                'dst_ip': alert.dst_ip,
                'protocol': alert.protocol,
                'label': alert.label,
                'severity': severity
            })
        
        return alert_data
    
    @database_sync_to_async
    def get_dashboard_stats(self):
        # Similar to dashboard_stats view
        end_time = timezone.now()
        start_time = end_time - datetime.timedelta(days=1)
        
        total_traffic = TrafficLog.objects.filter(timestamp__gte=start_time).count()
        
        alerts_count = TrafficLog.objects.filter(
            timestamp__gte=start_time,
            label__isnull=False
        ).exclude(label='').count()
        
        active_hosts = TrafficLog.objects.filter(
            timestamp__gte=start_time
        ).values('src_ip').distinct().count()
        
        data_transferred = TrafficLog.objects.filter(timestamp__gte=start_time).aggregate(
            total_bytes=models.Sum(models.F('totlen_fwd_pkts') + models.F('totlen_bwd_pkts'))
        )['total_bytes'] or 0
        
        data_transferred_gb = data_transferred / (1024 * 1024 * 1024)
        
        return {
            'total_traffic': total_traffic,
            'alerts_count': alerts_count,
            'active_hosts': active_hosts,
            'data_transferred': round(data_transferred_gb, 2)
        }
    
    # Broadcast method for new traffic logs
    async def traffic_update(self, event):
        """Handle new traffic log updates"""
        await self.send(text_data=json.dumps({
            'type': 'traffic_update',
            'data': event['data']
        }))
    
    # Broadcast method for new alerts
    async def alert_update(self, event):
        """Handle new alert updates"""
        await self.send(text_data=json.dumps({
            'type': 'alert_update',
            'data': event['data']
        }))