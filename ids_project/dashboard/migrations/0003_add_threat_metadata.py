from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("dashboard", "0002_ensure_dash_trafficlog_table"),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
                ALTER TABLE dash_trafficlog
                    ADD COLUMN IF NOT EXISTS threat_type VARCHAR(128) NULL,
                    ADD COLUMN IF NOT EXISTS threat_family VARCHAR(128) NULL,
                    ADD COLUMN IF NOT EXISTS threat_detail TEXT NULL;
            """,
            reverse_sql="""
                ALTER TABLE dash_trafficlog
                    DROP COLUMN IF EXISTS threat_type,
                    DROP COLUMN IF EXISTS threat_family,
                    DROP COLUMN IF EXISTS threat_detail;
            """,
        ),
    ]
