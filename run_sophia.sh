#!/bin/bash
# Sophia ML and Research System - Launch Script

# ANSI color codes for terminal output
BLUE="\033[94m"
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
BOLD="\033[1m"
RESET="\033[0m"

echo -e "${BLUE}${BOLD}Starting Sophia ML and Research System...${RESET}"

# Find Tekton root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ "$SCRIPT_DIR" == *"/utils" ]]; then
    # Script is running from a symlink in utils
    TEKTON_ROOT=$(cd "$SCRIPT_DIR" && cd "$(readlink "${BASH_SOURCE[0]}" | xargs dirname | xargs dirname)" && pwd)
else
    # Script is running from Sophia directory
    TEKTON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Ensure we're in the correct directory
cd "$SCRIPT_DIR"

# Set environment variables
export SOPHIA_PORT=8014
export PYTHONPATH="$SCRIPT_DIR:$TEKTON_ROOT:$PYTHONPATH"

# Create log directories
mkdir -p "$HOME/.tekton/logs"
mkdir -p "$HOME/.tekton/data/sophia"

# Error handling function
handle_error() {
    echo -e "${RED}Error: $1${RESET}" >&2
    ${TEKTON_ROOT}/scripts/tekton-register unregister --component sophia
    exit 1
}

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Register with Hermes using tekton-register
echo -e "${YELLOW}Registering Sophia with Hermes...${RESET}"
${TEKTON_ROOT}/scripts/tekton-register register --component sophia --config ${TEKTON_ROOT}/config/components/sophia.yaml &
REGISTER_PID=$!

# Give registration a moment to complete
sleep 2

# Start the Sophia service
echo -e "${YELLOW}Starting Sophia API server...${RESET}"
python -m sophia.api.app --port $SOPHIA_PORT > "$HOME/.tekton/logs/sophia.log" 2>&1 &
SOPHIA_PID=$!

# Trap signals for graceful shutdown
trap "${TEKTON_ROOT}/scripts/tekton-register unregister --component sophia; kill $SOPHIA_PID 2>/dev/null; exit" EXIT SIGINT SIGTERM

# Wait for the server to start
echo -e "${YELLOW}Waiting for Sophia to start...${RESET}"
for i in {1..30}; do
    if curl -s http://localhost:$SOPHIA_PORT/health >/dev/null; then
        echo -e "${GREEN}Sophia started successfully on port $SOPHIA_PORT${RESET}"
        echo -e "${GREEN}API available at: http://localhost:$SOPHIA_PORT/api${RESET}"
        break
    fi
    
    # Check if the process is still running
    if ! kill -0 $SOPHIA_PID 2>/dev/null; then
        echo -e "${RED}Sophia process terminated unexpectedly${RESET}"
        cat "$HOME/.tekton/logs/sophia.log"
        handle_error "Sophia failed to start"
    fi
    
    echo -n "."
    sleep 1
done

# Keep the script running to maintain registration
echo -e "${BLUE}Sophia is running. Press Ctrl+C to stop.${RESET}"
wait $SOPHIA_PID