#!/bin/bash
# TensorBoard Launch Script for Med-Framework
#
# This script launches TensorBoard to monitor training progress.
#
# Usage:
#   ./scripts/launch_tensorboard.sh [OPTIONS]
#
# Options:
#   --logdir DIR    Specify log directory (default: logs)
#   --port PORT     Specify port number (default: 6006)
#   --help          Show this help message

set -e

# Default configuration
LOGDIR="logs"
PORT=6006
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --logdir)
            LOGDIR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "TensorBoard Launch Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --logdir DIR    Specify log directory (default: logs)"
            echo "  --port PORT     Specify port number (default: 6006)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

# Check if log directory exists
if [ ! -d "$LOGDIR" ]; then
    echo -e "${YELLOW}Warning: Log directory '$LOGDIR' does not exist.${NC}"
    echo -e "${BLUE}Creating log directory...${NC}"
    mkdir -p "$LOGDIR"
    echo -e "${GREEN}✓ Log directory created: $LOGDIR${NC}"
fi

# Check if there are any log files
if [ -z "$(ls -A $LOGDIR 2>/dev/null)" ]; then
    echo -e "${YELLOW}Warning: Log directory is empty.${NC}"
    echo -e "${YELLOW}TensorBoard will start, but there are no logs to display yet.${NC}"
    echo -e "${YELLOW}Run a training session to generate logs.${NC}"
    echo ""
fi

# Check if TensorBoard is installed
if ! command -v tensorboard &> /dev/null; then
    echo -e "${RED}Error: TensorBoard is not installed.${NC}"
    echo -e "${BLUE}Installing TensorBoard...${NC}"
    uv add tensorboard
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install TensorBoard.${NC}"
        echo -e "${YELLOW}Please install manually: uv add tensorboard${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ TensorBoard installed successfully${NC}"
fi

# Print startup information
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Med-Framework TensorBoard Monitor${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Log Directory: ${YELLOW}$LOGDIR${NC}"
echo -e "  Port:          ${YELLOW}$PORT${NC}"
echo -e "  URL:           ${YELLOW}http://localhost:$PORT${NC}"
echo ""
echo -e "${BLUE}Starting TensorBoard...${NC}"
echo ""

# Launch TensorBoard
tensorboard --logdir="$LOGDIR" --port="$PORT" --bind_all

# Note: The script will not reach here unless TensorBoard is stopped
echo ""
echo -e "${GREEN}TensorBoard stopped.${NC}"
