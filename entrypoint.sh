#!/bin/sh
set -e

echo "Creating data directory if not exists..."
mkdir -p /app/data

echo "Running migrations..."
uv run alembic -c ./app/alembic.ini upgrade head

echo "Starting FastAPI server..."

CMD="uv run fastapi run"

if [ -n "$ROOT_PATH" ]; then
  CMD="$CMD --root-path $ROOT_PATH"
fi

exec $CMD