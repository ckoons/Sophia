#!/bin/bash
# Run the Sophia API server
# This script runs the Sophia API server with the specified configuration.

# Set default port
DEFAULT_PORT=8006
SOPHIA_PORT=${SOPHIA_PORT:-$DEFAULT_PORT}

# Set default host
DEFAULT_HOST="localhost"
SOPHIA_HOST=${SOPHIA_HOST:-$DEFAULT_HOST}

# Set default log level
DEFAULT_LOG_LEVEL="INFO"
SOPHIA_LOG_LEVEL=${SOPHIA_LOG_LEVEL:-$DEFAULT_LOG_LEVEL}

# Set path to this script and the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Output information about the environment
echo "Starting Sophia API server..."
echo "  Host: $SOPHIA_HOST"
echo "  Port: $SOPHIA_PORT"
echo "  Log Level: $SOPHIA_LOG_LEVEL"
echo "  Project Root: $PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Run the API server
echo "Starting API server..."
cd "$PROJECT_ROOT"

# Export environment variables
export SOPHIA_PORT=$SOPHIA_PORT
export SOPHIA_HOST=$SOPHIA_HOST
export SOPHIA_LOG_LEVEL=$SOPHIA_LOG_LEVEL

# Start the server
python -m sophia.api.app

# Exit with the same status as the API server
exit $?