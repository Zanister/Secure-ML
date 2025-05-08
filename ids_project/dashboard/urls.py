# urls.py (for the dashboard app)
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'traffic-logs', views.TrafficLogViewSet)

urlpatterns = [
    path('', views.index, name='dashboard_index'),
    path('api/', include(router.urls)),
    path('api/traffic-over-time/', views.traffic_over_time, name='traffic_over_time'),
    path('api/protocol-distribution/', views.protocol_distribution, name='protocol_distribution'),
    path('api/top-sources/', views.top_sources, name='top_sources'),
    path('api/recent-alerts/', views.recent_alerts, name='recent_alerts'),
    path('api/dashboard-stats/', views.dashboard_stats, name='dashboard_stats'),
]