# serializers.py
from rest_framework import serializers
from .models import TrafficLog

class TrafficLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrafficLog
        fields = '__all__'