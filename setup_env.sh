#!/bin/bash
set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "Model Runner Environment Setup"
echo "==============================="

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if module command exists
module_exists() {
    command_exists module
}

# Check if uv is installed
if ! command_exists uv; then
    echo -e "${RED}✗ uv not installed${NC}"
    echo "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Load modules only if module system is available
if module_exists; then
    # Check and load modules if available
    if module avail openssl/1.1.1 2>&1 | grep -q "openssl/1.1.1"; then
        module load openssl/1.1.1 2>/dev/null
    fi

    if module avail python/3.10.13 2>&1 | grep -q "python/3.10.13"; then
        module load python/3.10.13 2>/dev/null
    fi

    if module avail gcc/13.2.0 2>&1 | grep -q "gcc/13.2.0"; then
        module load gcc/13.2.0 2>/dev/null
    fi

    echo -e "${GREEN}✓ Modules loaded${NC}"
fi

# Sync dependencies
echo ""
if uv sync; then
    echo ""
    echo -e "${GREEN}✓ Setup complete${NC}"
    echo "Activate: source .venv/bin/activate"
else
    echo -e "${RED}✗ Sync failed${NC}"
    exit 1
fi