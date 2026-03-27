# Secure-ML

Secure-ML is a network intrusion detection system (IDS) pipeline with a live dashboard.

## What It Does

1. Captures network packets into `.pcap` files.
2. Converts PCAP to NetFlow-like feature CSVs using CICFlowMeter.
3. Preprocesses flow features and runs ML classification.
4. Saves results into PostgreSQL (`dash_trafficlog`).
5. Serves analytics and alerts through a Django + React dashboard.

## Project Layout

- `data_Capture/` packet capture, RabbitMQ producer/consumer
- `processing_analysis/` conversion, preprocessing, model inference
- `ids_project/` Django API + websocket + React dashboard
- `CICFlowMeter/` bundled flow feature extractor

## Configuration

Use environment variables for all runtime settings.

1. Copy `.env.example` to `.env`
2. Update values for your machine/services

Key variables:

- DB: `POSTGRES_*`
- RabbitMQ: `RABBITMQ_HOST`, `RABBITMQ_PORT`
- Capture: `CAPTURE_INTERFACE`, `CAPTURE_DURATION_SECONDS`
- Model: `MODEL_PATH`
- Fallback model: `FALLBACK_MODEL_PATH`, `MODEL_BOOTSTRAP_CSV`
- Channels: `USE_REDIS` (`false` for simple local start)

## Quickstart (Docker-Only)

No local Python/Node/Postgres/RabbitMQ installs are required.

1. Install Docker Desktop.
2. (Optional) Copy `.env.example` to `.env` and tweak values.
3. Start the stack:
   - `docker compose up --build`
4. Open dashboard:
   - `http://localhost:8000`

Included services:

- `web`: Django app + API + websocket server
- `worker`: RabbitMQ consumer + processing pipeline
- `capture`: tcpdump producer that captures interface traffic and publishes PCAP paths
- `postgres`: persistent database
- `rabbitmq`: message broker + management UI (`http://localhost:15672`)
- `redis`: Channels backend

## Notes

- If `MODEL_PATH` points to a valid neural model, `receive.py` uses it.
- If the neural model is missing, worker startup bootstraps a hybrid sklearn stack from `MODEL_BOOTSTRAP_CSV`:
  - supervised classifier at `FALLBACK_MODEL_PATH` (`baseline-ids.joblib`)
  - anomaly detector at `ANOMALY_MODEL_PATH` (`baseline-anomaly.joblib`)
  This catches both known classes and unknown flow anomalies.
- If both models are unavailable, `receive.py` falls back to rule-based threat labeling to preserve IDS continuity.
- Every alert now carries analyst metadata: `detection_source` and `confidence` for better triage decisions.
- WebSocket clients auto-reconnect, and worker broadcasts alerts directly to Channels so realtime updates continue even when inserts happen via raw SQL.
- `capture` and `worker` share a `/captures` Docker volume so generated PCAP files are readable by the classifier pipeline.
- For POC on Docker Desktop, capture runs fully in-container on the compose network interface (`eth0`) with no host networking dependency.
- This mode is excellent for pipeline validation and realtime dashboard behavior.
- Frontend assets are built during Docker image build (Node runs in the build stage only).
- For production, secure credentials and restrict exposed ports.
