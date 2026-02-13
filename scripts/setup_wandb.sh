#!/bin/bash
# Weights & Biases (WandB) Setup Script for Med-Framework
#
# This script helps configure Weights & Biases for experiment tracking.
#
# Usage:
#   ./scripts/setup_wandb.sh [OPTIONS]
#
# Options:
#   --api-key KEY   Provide WandB API key directly
#   --project NAME  Set default project name (default: med-framework)
#   --entity NAME   Set WandB entity/team name
#   --help          Show this help message

set -e

# Default configuration
PROJECT_NAME="med-framework"
ENTITY=""
API_KEY=""
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
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --entity)
            ENTITY="$2"
            shift 2
            ;;
        --help)
            echo "Weights & Biases Setup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --api-key KEY   Provide WandB API key directly"
            echo "  --project NAME  Set default project name (default: med-framework)"
            echo "  --entity NAME   Set WandB entity/team name"
            echo "  --help          Show this help message"
            echo ""
            echo "To get your API key:"
            echo "  1. Sign up at https://wandb.ai"
            echo "  2. Go to https://wandb.ai/authorize"
            echo "  3. Copy your API key"
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

# Print header
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Med-Framework WandB Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if wandb is installed
echo -e "${BLUE}Checking WandB installation...${NC}"
if ! command -v wandb &> /dev/null; then
    echo -e "${YELLOW}WandB is not installed.${NC}"
    echo -e "${BLUE}Installing WandB...${NC}"
    uv add wandb
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install WandB.${NC}"
        echo -e "${YELLOW}Please install manually: uv add wandb${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ WandB installed successfully${NC}"
else
    echo -e "${GREEN}✓ WandB is already installed${NC}"
fi

echo ""

# Login to WandB
echo -e "${BLUE}Configuring WandB authentication...${NC}"
echo ""

if [ -n "$API_KEY" ]; then
    # Use provided API key
    echo -e "${BLUE}Using provided API key...${NC}"
    wandb login "$API_KEY"
else
    # Check if already logged in
    if wandb whoami &> /dev/null; then
        echo -e "${GREEN}✓ Already logged in to WandB${NC}"
        CURRENT_USER=$(wandb whoami 2>/dev/null | grep -oP '(?<=Logged in as: ).*' || echo "unknown")
        echo -e "${BLUE}Current user: ${YELLOW}$CURRENT_USER${NC}"
        echo ""
        read -p "Do you want to re-login? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Keeping current login.${NC}"
        else
            wandb login
        fi
    else
        # Interactive login
        echo -e "${YELLOW}Please login to WandB.${NC}"
        echo -e "${BLUE}You will be redirected to get your API key.${NC}"
        echo ""
        wandb login
    fi
fi

echo ""

# Create WandB configuration file
echo -e "${BLUE}Creating WandB configuration...${NC}"

WANDB_CONFIG_DIR="$PROJECT_ROOT/.wandb"
mkdir -p "$WANDB_CONFIG_DIR"

# Create settings file
SETTINGS_FILE="$WANDB_CONFIG_DIR/settings"
cat > "$SETTINGS_FILE" << EOF
[default]
project = $PROJECT_NAME
EOF

if [ -n "$ENTITY" ]; then
    echo "entity = $ENTITY" >> "$SETTINGS_FILE"
fi

echo -e "${GREEN}✓ Configuration saved to $SETTINGS_FILE${NC}"

# Create example configuration for Python
EXAMPLE_CONFIG="$PROJECT_ROOT/configs/wandb_config.yaml"
echo -e "${BLUE}Creating example configuration...${NC}"

cat > "$EXAMPLE_CONFIG" << 'EOF'
# Weights & Biases Configuration Example
#
# This file shows how to configure WandB in your training scripts.
# Copy relevant sections to your main config file.

logging:
  # Enable WandB logging
  use_wandb: true

  # WandB project settings
  wandb_project: "med-framework"
  wandb_entity: null  # Set to your team name if using WandB teams

  # Experiment naming
  wandb_run_name: null  # Auto-generated if null
  wandb_tags: ["medical", "multimodal"]

  # What to log
  wandb_log_model: true  # Save model checkpoints to WandB
  wandb_log_gradients: false  # Log gradient histograms (can be slow)
  wandb_log_images: true  # Log sample images
  wandb_watch_model: true  # Watch model architecture and gradients

  # Logging frequency
  wandb_log_interval: 10  # Log every N steps

# Example: How to use in Python code
#
# import wandb
# from med_core.configs import load_config
#
# config = load_config("configs/your_config.yaml")
#
# # Initialize WandB
# if config.logging.use_wandb:
#     wandb.init(
#         project=config.logging.wandb_project,
#         entity=config.logging.wandb_entity,
#         name=config.logging.wandb_run_name,
#         tags=config.logging.wandb_tags,
#         config=config.to_dict()
#     )
#
# # During training
# if config.logging.use_wandb:
#     wandb.log({
#         "train/loss": loss,
#         "train/accuracy": acc,
#         "epoch": epoch
#     })
#
# # Save model
# if config.logging.use_wandb and config.logging.wandb_log_model:
#     wandb.save("model.pth")
EOF

echo -e "${GREEN}✓ Example configuration saved to $EXAMPLE_CONFIG${NC}"

# Create quick start guide
GUIDE_FILE="$PROJECT_ROOT/docs/wandb_guide.md"
echo -e "${BLUE}Creating quick start guide...${NC}"

cat > "$GUIDE_FILE" << 'EOF'
# Weights & Biases Integration Guide

## Overview

Weights & Biases (WandB) is a powerful experiment tracking and visualization platform for machine learning projects. This guide shows how to use WandB with the Med-Framework.

## Setup

### 1. Install and Configure

Run the setup script:

```bash
./scripts/setup_wandb.sh
```

Or with options:

```bash
./scripts/setup_wandb.sh --project my-medical-project --entity my-team
```

### 2. Get Your API Key

1. Sign up at [https://wandb.ai](https://wandb.ai)
2. Go to [https://wandb.ai/authorize](https://wandb.ai/authorize)
3. Copy your API key

## Usage

### Basic Configuration

Add to your config YAML file:

```yaml
logging:
  use_wandb: true
  wandb_project: "med-framework"
  wandb_entity: null  # Your team name (optional)
  wandb_tags: ["medical", "multimodal"]
```

### In Training Scripts

```python
import wandb
from med_core.configs import load_config

# Load configuration
config = load_config("configs/your_config.yaml")

# Initialize WandB
if config.logging.use_wandb:
    wandb.init(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        config=config.to_dict()
    )

# Log metrics during training
for epoch in range(num_epochs):
    # ... training code ...

    if config.logging.use_wandb:
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": current_lr
        })

# Log images
if config.logging.use_wandb:
    wandb.log({
        "examples": [wandb.Image(img) for img in sample_images]
    })

# Save model
if config.logging.use_wandb:
    wandb.save("checkpoints/best_model.pth")

# Finish run
if config.logging.use_wandb:
    wandb.finish()
```

## Features

### 1. Experiment Tracking

- Automatic logging of hyperparameters
- Real-time metric visualization
- Compare multiple runs
- Filter and search experiments

### 2. Model Versioning

- Save model checkpoints
- Track model lineage
- Download models from any run

### 3. Visualization

- Interactive plots
- Custom charts
- Image and media logging
- Confusion matrices

### 4. Collaboration

- Share experiments with team
- Add notes and comments
- Create reports

## Best Practices

### 1. Naming Conventions

Use descriptive run names:

```python
run_name = f"{model_name}_{dataset}_{timestamp}"
wandb.init(name=run_name)
```

### 2. Tagging

Tag experiments for easy filtering:

```python
tags = ["resnet50", "multimodal", "experiment-v2"]
wandb.init(tags=tags)
```

### 3. Grouping

Group related runs:

```python
wandb.init(group="hyperparameter-search", job_type="train")
```

### 4. Logging Frequency

Balance detail vs performance:

```python
# Log every N steps
if step % log_interval == 0:
    wandb.log(metrics)
```

## Comparison: TensorBoard vs WandB

| Feature | TensorBoard | WandB |
|---------|-------------|-------|
| Setup | Local only | Cloud-based |
| Collaboration | Limited | Excellent |
| Model versioning | No | Yes |
| Hyperparameter tracking | Basic | Advanced |
| Cost | Free | Free tier + paid |
| Offline mode | Yes | Yes |

## Troubleshooting

### Issue: Login fails

**Solution:**
```bash
# Re-login
wandb login --relogin

# Or set API key directly
export WANDB_API_KEY=your_key_here
```

### Issue: Slow logging

**Solution:**
```python
# Reduce logging frequency
wandb_log_interval: 100  # Instead of 10

# Disable gradient logging
wandb_log_gradients: false
```

### Issue: Offline mode

**Solution:**
```bash
# Enable offline mode
export WANDB_MODE=offline

# Later, sync offline runs
wandb sync wandb/offline-run-*
```

## Environment Variables

```bash
# API key
export WANDB_API_KEY=your_key

# Project name
export WANDB_PROJECT=med-framework

# Entity/team
export WANDB_ENTITY=your_team

# Offline mode
export WANDB_MODE=offline

# Disable WandB
export WANDB_DISABLED=true
```

## Resources

- [WandB Documentation](https://docs.wandb.ai)
- [WandB Python Library](https://github.com/wandb/wandb)
- [Example Projects](https://wandb.ai/gallery)

## Support

For issues or questions:
- WandB Community: [https://community.wandb.ai](https://community.wandb.ai)
- GitHub Issues: [https://github.com/wandb/wandb/issues](https://github.com/wandb/wandb/issues)
EOF

echo -e "${GREEN}✓ Guide saved to $GUIDE_FILE${NC}"

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Project: ${YELLOW}$PROJECT_NAME${NC}"
if [ -n "$ENTITY" ]; then
    echo -e "  Entity:  ${YELLOW}$ENTITY${NC}"
fi
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  1. Update your config YAML:"
echo -e "     ${YELLOW}logging.use_wandb: true${NC}"
echo -e "  2. Run training:"
echo -e "     ${YELLOW}uv run med-train --config configs/your_config.yaml${NC}"
echo -e "  3. View results:"
echo -e "     ${YELLOW}https://wandb.ai/$PROJECT_NAME${NC}"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo -e "  - Quick Start: ${YELLOW}docs/wandb_guide.md${NC}"
echo -e "  - Example Config: ${YELLOW}configs/wandb_config.yaml${NC}"
echo ""
echo -e "${GREEN}Happy experimenting! 🚀${NC}"
echo ""
