#!/bin/bash

# Sophia Setup Script
# Creates a virtual environment and installs dependencies for Sophia

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ANSI color codes
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo -e "${GREEN}Setting up Sophia...${NC}"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed or not in the PATH${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Sophia in development mode
echo -e "${YELLOW}Installing Sophia and dependencies...${NC}"
pip install -e .

# Install optional ML dependencies if available
if pip install torch torchvision --quiet; then
    echo -e "${GREEN}Installed PyTorch${NC}"
else
    echo -e "${YELLOW}Could not install PyTorch. Using minimal ML dependencies.${NC}"
    pip install scikit-learn
fi

# Install development dependencies
pip install pytest pytest-asyncio

# Create data directory
mkdir -p ~/.tekton/data/sophia

echo -e "${GREEN}Sophia setup complete!${NC}"
echo -e "To activate the virtual environment, run:\n  source venv/bin/activate\n"
