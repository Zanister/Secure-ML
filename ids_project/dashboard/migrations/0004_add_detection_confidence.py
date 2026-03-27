from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("dashboard", "0003_add_threat_metadata"),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
                ALTER TABLE dash_trafficlog
                    ADD COLUMN IF NOT EXISTS detection_source VARCHAR(64) NULL,
                    ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NULL;
            """,
            reverse_sql="""
                ALTER TABLE dash_trafficlog
                    DROP COLUMN IF EXISTS detection_source,
                    DROP COLUMN IF EXISTS confidence;
            """,
        ),
    ]
