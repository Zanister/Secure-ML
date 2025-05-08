# models.py
from django.db import models

# This model will map to your existing PostgreSQL table
class TrafficLog(models.Model):
    timestamp = models.DateTimeField()
    src_ip = models.CharField(max_length=100, null=True, blank=True)
    dst_ip = models.CharField(max_length=100, null=True, blank=True)
    src_port = models.IntegerField(null=True, blank=True)
    dst_port = models.IntegerField(null=True, blank=True)
    protocol = models.CharField(max_length=50, null=True, blank=True)
    label = models.CharField(max_length=100, null=True, blank=True)
    flow_duration = models.BigIntegerField(null=True, blank=True)
    tot_fwd_pkts = models.IntegerField(null=True, blank=True)
    tot_bwd_pkts = models.IntegerField(null=True, blank=True)
    fwd_pkts_per_sec = models.FloatField(null=True, blank=True)
    bwd_pkts_per_sec = models.FloatField(null=True, blank=True)
    flow_byts_per_sec = models.FloatField(null=True, blank=True)
    flow_pkts_per_sec = models.FloatField(null=True, blank=True)
    flow_iat_mean = models.FloatField(null=True, blank=True)
    flow_iat_std = models.FloatField(null=True, blank=True)
    totlen_fwd_pkts = models.FloatField(null=True, blank=True)
    totlen_bwd_pkts = models.FloatField(null=True, blank=True)
    fwd_pkt_len_max = models.FloatField(null=True, blank=True)
    fwd_pkt_len_min = models.FloatField(null=True, blank=True)
    fwd_pkt_len_mean = models.FloatField(null=True, blank=True)
    fwd_pkt_len_std = models.FloatField(null=True, blank=True)
    bwd_pkt_len_max = models.FloatField(null=True, blank=True)
    bwd_pkt_len_min = models.FloatField(null=True, blank=True)
    bwd_pkt_len_mean = models.FloatField(null=True, blank=True)
    bwd_pkt_len_std = models.FloatField(null=True, blank=True)
    flow_iat_max = models.FloatField(null=True, blank=True)
    flow_iat_min = models.FloatField(null=True, blank=True)
    fwd_iat_tot = models.FloatField(null=True, blank=True)
    fwd_iat_mean = models.FloatField(null=True, blank=True)
    fwd_iat_std = models.FloatField(null=True, blank=True)
    fwd_iat_max = models.FloatField(null=True, blank=True)
    fwd_iat_min = models.FloatField(null=True, blank=True)

    class Meta:
        # This is important - tells Django this is an existing table
        db_table = 'dash_trafficlog'
        managed = False  # Django won't try to create this table

    def __str__(self):
        return f"{self.src_ip} → {self.dst_ip} ({self.protocol}) at {self.timestamp}"