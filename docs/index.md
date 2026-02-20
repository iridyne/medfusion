# MedFusion Documentation

Welcome to MedFusion's documentation!

MedFusion is a comprehensive framework for multimodal medical imaging analysis, 
providing state-of-the-art fusion techniques for combining different imaging modalities 
and clinical data.

## Features

- **Multimodal Fusion**: Advanced fusion strategies for combining medical imaging modalities
- **Attention Mechanisms**: Built-in attention supervision and interpretability
- **Multiple Instance Learning**: Support for MIL aggregation strategies
- **Flexible Architecture**: Pluggable backbones, fusion modules, and classification heads
- **Production Ready**: Docker support, CI/CD, comprehensive testing

## Quick Links

```{toctree}
:maxdepth: 2
:caption: User Guide

guides/quick_reference
guides/faq_troubleshooting
guides/docker_deployment
guides/ci_cd
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/med_core
api/aggregators
api/attention_supervision
api/backbones
api/datasets
api/evaluation
api/fusion
api/heads
api/models
api/preprocessing
api/trainers
api/utils
```

```{toctree}
:maxdepth: 1
:caption: Reference

reference/framework_error_codes
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medfusion.git
cd medfusion

# Install dependencies
pip install -e .

# Or use Docker
docker-compose up medfusion-train
```

## Quick Start

```python
from med_core.models import create_model
from med_core.datasets import create_dataset
from med_core.trainers import create_trainer

# Create model
model = create_model('multimodal_fusion', config)

# Create dataset
train_dataset = create_dataset('multiview', config)

# Create trainer
trainer = create_trainer('classification', model, config)

# Train
trainer.train(train_dataset)
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
