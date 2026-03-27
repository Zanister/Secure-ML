from django.db import migrations


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dash_trafficlog (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    src_ip TEXT NULL,
    dst_ip TEXT NULL,
    src_port INTEGER NULL,
    dst_port INTEGER NULL,
    protocol TEXT NULL,
    label TEXT NULL,
    flow_duration BIGINT NULL,
    tot_fwd_pkts INTEGER NULL,
    tot_bwd_pkts INTEGER NULL,
    fwd_pkts_per_sec DOUBLE PRECISION NULL,
    bwd_pkts_per_sec DOUBLE PRECISION NULL,
    flow_byts_per_sec DOUBLE PRECISION NULL,
    flow_pkts_per_sec DOUBLE PRECISION NULL,
    flow_iat_mean DOUBLE PRECISION NULL,
    flow_iat_std DOUBLE PRECISION NULL,
    totlen_fwd_pkts DOUBLE PRECISION NULL,
    totlen_bwd_pkts DOUBLE PRECISION NULL,
    fwd_pkt_len_max DOUBLE PRECISION NULL,
    fwd_pkt_len_min DOUBLE PRECISION NULL,
    fwd_pkt_len_mean DOUBLE PRECISION NULL,
    fwd_pkt_len_std DOUBLE PRECISION NULL,
    bwd_pkt_len_max DOUBLE PRECISION NULL,
    bwd_pkt_len_min DOUBLE PRECISION NULL,
    bwd_pkt_len_mean DOUBLE PRECISION NULL,
    bwd_pkt_len_std DOUBLE PRECISION NULL,
    flow_iat_max DOUBLE PRECISION NULL,
    flow_iat_min DOUBLE PRECISION NULL,
    fwd_iat_tot DOUBLE PRECISION NULL,
    fwd_iat_mean DOUBLE PRECISION NULL,
    fwd_iat_std DOUBLE PRECISION NULL,
    fwd_iat_max DOUBLE PRECISION NULL,
    fwd_iat_min DOUBLE PRECISION NULL
);
"""


DROP_TABLE_SQL = "DROP TABLE IF EXISTS dash_trafficlog;"


class Migration(migrations.Migration):
    dependencies = [
        ("dashboard", "0001_initial"),
    ]

    operations = [
        migrations.RunSQL(
            sql=CREATE_TABLE_SQL,
            reverse_sql=DROP_TABLE_SQL,
        ),
    ]
