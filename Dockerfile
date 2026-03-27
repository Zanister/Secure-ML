FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /build
COPY ids_project/package*.json ./ids_project/
COPY ids_project/webpack.config.js ./ids_project/
COPY ids_project/frontend/src ./ids_project/frontend/src
RUN cd ids_project && npm ci && npx webpack --config webpack.config.js

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    netcat-openbsd \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-builder /build/ids_project/dashboard/static/js/dashboard.js /app/ids_project/dashboard/static/js/dashboard.js

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "ids_project/manage.py", "runserver", "0.0.0.0:8000"]