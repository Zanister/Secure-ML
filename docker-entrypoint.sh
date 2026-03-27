#!/bin/sh
set -eu

DB_HOST="${POSTGRES_HOST:-postgres}"
DB_PORT="${POSTGRES_PORT:-5432}"
RMQ_HOST="${RABBITMQ_HOST:-rabbitmq}"
RMQ_PORT="${RABBITMQ_PORT:-5672}"

echo "Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT}..."
until nc -z "${DB_HOST}" "${DB_PORT}"; do
  sleep 2
done

echo "Waiting for RabbitMQ at ${RMQ_HOST}:${RMQ_PORT}..."
until nc -z "${RMQ_HOST}" "${RMQ_PORT}"; do
  sleep 2
done

exec "$@"
