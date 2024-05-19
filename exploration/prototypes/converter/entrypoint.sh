#!/bin/bash

set -e

exec python /app/composites/calculator.py &
exec python /app/composites/converter_api.py
