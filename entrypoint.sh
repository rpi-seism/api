#!/bin/sh
set -e

echo "Creating data directory if not exists..."
mkdir -p /data

echo "Running migrations..."
uv run alembic -c ./app/alembic.ini upgrade head

echo "Starting FastAPI server..."

CMD="uv run fastapi run main.py --port 8000"

if [ -n "$ROOT_PATH" ]; then
  CMD="$CMD --root-path $ROOT_PATH"
fi

exec $CMD