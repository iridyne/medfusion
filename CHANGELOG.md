# Changelog

All notable changes to MedFusion will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-20

### Breaking Changes

- **Removed deprecated `med_core.configs.attention_config` module**
  - All attention supervision configuration is now integrated into `ExperimentConfig`
  - See [Migration Guide](docs/guides/migration_attention_config.md) for details
  - Classes removed:
    - `AttentionSupervisionConfig`
    - `ExperimentConfigWithAttention`
    - `DataConfigWithMask`
    - `TrainingConfigWithAttention`
  - Functions removed:
    - `create_mask_supervised_config()`
    - `create_cam_supervised_config()`
    - `create_mil_config()`
    - `create_bbox_supervised_config()`

### Added

- **Comprehensive API Documentation System**
  - Sphinx-based documentation with Read the Docs theme
  - 12 API reference pages covering all modules
  - Automatic generation from docstrings
  - Build script: `./scripts/build_docs.sh`
  - Documentation guide: `docs/guides/api_documentation.md`

- **Enhanced Test Coverage**
  - 70+ new tests for aggregators and heads modules
  - Test coverage analysis script: `scripts/analyze_coverage.py`
  - Test stub generator: `scripts/generate_test_stubs.py`
  - Coverage documentation: `docs/TEST_COVERAGE_IMPROVEMENT.md`

- **Configuration Validation System**
  - Comprehensive validation with 30+ error codes
  - Helpful error messages with suggestions
  - Demo script: `scripts/demo_validation.py`
  - 11 validation tests

- **Enhanced Error Handling**
  - Structured error codes (E000-E1000+)
  - Context information and recovery suggestions
  - 23 error handling tests
  - Demo script: `scripts/demo_error_handling.py`

- **Improved Logging System**
  - Structured logging with context
  - JSON format support
  - Performance tracking
  - Metrics logging
  - 16 logging tests
  - Demo script: `scripts/demo_logging.py`

- **Docker Support**
  - Multi-service Docker Compose setup
  - Services: train, eval, tensorboard, jupyter, dev
  - Optimized Dockerfile with layer caching
  - Comprehensive deployment guide: `docs/guides/docker_deployment.md`

- **CI/CD Pipeline**
  - GitHub Actions workflows:
    - `ci.yml`: Continuous integration with tests
    - `release.yml`: Automated releases
    - `code-quality.yml`: Code quality checks
  - Pre-commit hooks configuration
  - CI/CD guide: `docs/guides/ci_cd.md`

- **Documentation**
  - FAQ and troubleshooting guide: `docs/guides/faq_troubleshooting.md`
  - Quick reference guide: `docs/guides/quick_reference.md`
  - Framework error codes reference: `docs/reference/framework_error_codes.md`
  - Migration guide: `docs/guides/migration_attention_config.md`

### Changed

- Updated version from 0.1.0 to 0.2.0
- Improved project structure and organization
- Enhanced documentation throughout codebase

### Deprecated

- None (deprecated items from 0.1.x have been removed)

### Removed

- `med_core.configs.attention_config` module (see Breaking Changes)

### Fixed

- Various bug fixes and improvements
- Enhanced error messages and validation

### Security

- No security updates in this release

## [0.1.0] - 2026-01-15

### Added

- Initial release of MedFusion framework
- Core multimodal fusion functionality
- Attention supervision mechanisms
- Multiple Instance Learning (MIL) aggregators
- Various backbone architectures
- Dataset implementations
- Training and evaluation pipelines
- Basic documentation

### Deprecated

- `med_core.configs.attention_config` module (deprecated with warnings)
  - Scheduled for removal in 0.2.0

---

## Migration Guides

- [0.1.x → 0.2.0: attention_config Removal](docs/guides/migration_attention_config.md)

## Links

- [Documentation](docs/README.md)
- [API Reference](docs/api/med_core.md)
- [GitHub Repository](https://github.com/yourusername/medfusion)
- [Issue Tracker](https://github.com/yourusername/medfusion/issues)

## Version Support

| Version | Status | Support Until | Python | PyTorch |
|---------|--------|---------------|--------|---------|
| 0.2.x   | Active | TBD           | 3.11+  | 2.0+    |
| 0.1.x   | Deprecated | 2026-06-20 | 3.11+  | 2.0+    |

## Upgrade Instructions

### From 0.1.x to 0.2.0

1. **Update package:**
   ```bash
   pip install --upgrade medfusion
   ```

2. **Migrate configuration:**
   - Follow the [Migration Guide](docs/guides/migration_attention_config.md)
   - Update imports from `attention_config` to `ExperimentConfig`
   - Update configuration attribute names

3. **Test your code:**
   ```bash
   pytest tests/
   ```

4. **Update documentation:**
   - Review new API documentation
   - Check updated examples

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to MedFusion.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
