# med_core Package

The core package containing all MedFusion functionality.

## Submodules

- [aggregators](/contents/api/aggregators) - Multiple Instance Learning aggregators
- [attention_supervision](/contents/api/attention_supervision) - Attention supervision mechanisms
- [backbones](/contents/api/backbones) - Feature extraction backbones
- [datasets](/contents/api/datasets) - Dataset implementations
- [evaluation](/contents/api/evaluation) - Evaluation metrics and tools
- [fusion](/contents/api/fusion) - Multimodal fusion strategies
- [heads](/contents/api/heads) - Classification and prediction heads
- [models](/contents/api/models) - Complete model implementations
- [preprocessing](/contents/api/preprocessing) - Data preprocessing utilities
- [trainers](/contents/api/trainers) - Training loops and strategies
- [utils](/contents/api/utils) - Utility functions
- `med_core.output_layout` - Shared run / seed / stability output directory helpers
- `med_core.stability` - Reusable multi-seed stability study runner and summary writers

## Configuration

The `med_core.configs` module provides configuration management and validation for MedFusion experiments.

### Configuration Validation

Configuration validation ensures all settings are correct before training begins.
