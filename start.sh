#!/bin/bash

echo "Waiting for RabbitMQ and PostgreSQL to be ready..."

until nc -z postgres 5432 && nc -z rabbitmq 5672; do
  echo "Waiting for services..."
  sleep 2
done

echo "Services are up!"

echo "Running send.py..."
python3 data_Capture/send.py

echo "Running receive.py..."
python3 data_Capture/receive.py
