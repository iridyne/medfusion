# Medical Multimodal System - Error Codes and Guardrails

**Version:** 1.0.0  
**Last Updated:** 2026-01-28  
**Purpose:** Define error handling, fallback strategies, and user feedback for medical data quality issues

---

## Overview

Medical data is inherently unstable and incomplete. This document defines a comprehensive error handling system that:

1. **Detects** data quality issues before model inference
2. **Degrades gracefully** when modalities are missing
3. **Provides professional feedback** to clinicians instead of system crashes
4. **Logs** all quality issues for audit and improvement

---

## Error Code Structure

Format: `MED-[CATEGORY]-[NUMBER]`

- **CATEGORY**: DATA, MODEL, SYSTEM, VALIDATION
- **NUMBER**: 3-digit sequential identifier

---

## 1. Data Quality Errors (MED-DATA-xxx)

### MED-DATA-001: Missing Image Modality

**Severity:** WARNING  
**Trigger:** Image file path exists but file is missing or corrupted  
**System Behavior:** Degrade to tabular-only mode

**User Message:**
```
⚠️ 影像数据缺失警告

检测到患者影像文件缺失或无法读取。系统将基于临床指标进行分析。

建议：
- 请确认影像文件路径是否正确
- 如需完整分析，请上传影像数据后重新提交

当前分析模式：仅临床指标分析
置信度：降低约 15-25%
```

**Technical Details:**
- Fallback: Use `TabularBranch` only
- Log: Record missing file path and patient ID
- Confidence penalty: -15% to -25%

---

### MED-DATA-002: Missing Tabular Modality

**Severity:** WARNING  
**Trigger:** Clinical features are all null or missing  
**System Behavior:** Degrade to vision-only mode

**User Message:**
```
⚠️ 临床数据缺失警告

检测到患者临床指标数据不完整。系统将仅基于影像进行分析。

建议：
- 请补充患者基本信息（年龄、性别等）
- 请补充生化指标数据以提高分析准确性

当前分析模式：仅影像分析
置信度：降低约 10-20%
```

**Technical Details:**
- Fallback: Use `VisionBranch` only
- Log: Record missing feature columns
- Confidence penalty: -10% to -20%

---

### MED-DATA-003: Both Modalities Missing

**Severity:** ERROR  
**Trigger:** Both image and tabular data are unavailable  
**System Behavior:** Abort inference, return error

**User Message:**
```
❌ 数据不足错误

无法进行分析：影像数据和临床数据均缺失。

要求：
- 至少需要提供影像数据或完整的临床指标数据
- 建议同时提供两种数据以获得最佳分析结果

请补充数据后重新提交。
```

**Technical Details:**
- Action: Raise `InsufficientDataError`
- Log: Record patient ID and data source
- No inference performed

---

### MED-DATA-004: Image Quality Too Low

**Severity:** WARNING  
**Trigger:** Image resolution < 128x128 or severe artifacts detected  
**System Behavior:** Continue with warning, flag low confidence

**User Message:**
```
⚠️ 影像质量警告

检测到影像质量较低，可能影响分析准确性。

问题：
- 分辨率过低（当前：{width}x{height}，建议：≥224x224）
- 或检测到严重伪影

建议：
- 如可能，请提供更高质量的影像
- 分析结果仅供参考，建议结合临床判断

当前分析：继续进行，但置信度降低
```

**Technical Details:**
- Continue inference with quality flag
- Log: Image dimensions and quality score
- Confidence penalty: -20% to -40%

---

### MED-DATA-005: Feature Out of Valid Range

**Severity:** WARNING  
**Trigger:** Clinical feature value outside expected physiological range  
**System Behavior:** Clip to valid range and continue

**User Message:**
```
⚠️ 数据异常警告

检测到以下临床指标超出正常生理范围：

- {feature_name}: {value} {unit} (正常范围: {min}-{max})

处理：
- 系统已自动将异常值调整至合理范围
- 建议核实数据录入是否正确

分析将继续进行，但请注意数据质量。
```

**Technical Details:**
- Action: Clip value to `[min, max]` from data dictionary
- Log: Original value, clipped value, feature name
- Flag: Mark as data quality issue

---

### MED-DATA-006: Incomplete Image Series

**Severity:** WARNING  
**Trigger:** CT/MRI series has < 50% expected slices  
**System Behavior:** Use available slices with warning

**User Message:**
```
⚠️ 影像序列不完整

检测到影像序列缺失部分切片。

当前状态：
- 预期切片数：{expected_slices}
- 实际切片数：{actual_slices}
- 完整度：{completeness}%

建议：
- 如可能，请提供完整的影像序列
- 当前分析基于可用切片，结果可能不完整

分析将继续，但置信度降低。
```

**Technical Details:**
- Use available slices
- Log: Expected vs actual slice count
- Confidence penalty: Proportional to missing percentage

---

### MED-DATA-007: DICOM Metadata Missing

**Severity:** INFO  
**Trigger:** DICOM file lacks critical metadata (StudyDate, Modality, etc.)  
**System Behavior:** Continue with default assumptions

**User Message:**
```
ℹ️ 影像元数据缺失

DICOM 文件缺少部分元数据信息。

缺失项：
- {missing_fields}

影响：
- 无法进行时序分析
- 无法验证影像模态

系统将使用默认设置继续分析。
```

**Technical Details:**
- Use default values for missing metadata
- Log: Missing metadata fields
- No confidence penalty

---

## 2. Model Errors (MED-MODEL-xxx)

### MED-MODEL-001: Model Checkpoint Not Found

**Severity:** ERROR  
**Trigger:** Model weights file missing or corrupted  
**System Behavior:** Abort, cannot proceed

**User Message:**
```
❌ 模型加载失败

系统无法加载分析模型。

错误原因：
- 模型文件缺失或损坏
- 路径：{checkpoint_path}

请联系技术支持团队。
```

**Technical Details:**
- Action: Raise `ModelLoadError`
- Log: Checkpoint path, error traceback
- Notify: System administrator

---

### MED-MODEL-002: Model Inference Timeout

**Severity:** ERROR  
**Trigger:** Inference takes > 60 seconds  
**System Behavior:** Abort inference, return timeout error

**User Message:**
```
❌ 分析超时

模型分析时间超过预期，已自动终止。

可能原因：
- 影像数据过大
- 系统负载过高

建议：
- 请稍后重试
- 如问题持续，请联系技术支持
```

**Technical Details:**
- Action: Kill inference process
- Log: Inference time, data size, system load
- Retry: Allow user to retry with timeout extension

---

### MED-MODEL-003: Low Confidence Prediction

**Severity:** WARNING  
**Trigger:** Model confidence < 60%  
**System Behavior:** Return result with strong warning

**User Message:**
```
⚠️ 低置信度预测

模型分析完成，但置信度较低。

预测结果：{prediction}
置信度：{confidence}%

重要提示：
- 此结果仅供参考，不应作为诊断依据
- 建议结合临床经验和其他检查结果综合判断
- 如有疑问，建议进行进一步检查

请谨慎使用此分析结果。
```

**Technical Details:**
- Return prediction with warning flag
- Log: Confidence score, prediction, patient ID
- Recommend: Manual review

---

### MED-MODEL-004: Conflicting Modality Predictions

**Severity:** WARNING  
**Trigger:** Vision and tabular branches predict different classes  
**System Behavior:** Return fusion result with conflict warning

**User Message:**
```
⚠️ 多模态预测不一致

检测到影像分析和临床指标分析结果存在差异。

影像分析：{vision_prediction} (置信度: {vision_conf}%)
临床分析：{tabular_prediction} (置信度: {tabular_conf}%)
综合分析：{fusion_prediction} (置信度: {fusion_conf}%)

建议：
- 请仔细核对影像和临床数据的一致性
- 建议进行进一步检查以明确诊断
- 此情况可能提示复杂病例，建议专家会诊

请谨慎解读分析结果。
```

**Technical Details:**
- Return all three predictions
- Log: Vision, tabular, fusion predictions
- Flag: Mark for expert review

---

## 3. Validation Errors (MED-VALIDATION-xxx)

### MED-VALIDATION-001: Patient ID Missing

**Severity:** ERROR  
**Trigger:** Patient identifier not provided  
**System Behavior:** Abort, cannot proceed

**User Message:**
```
❌ 患者信息缺失

无法进行分析：缺少患者标识信息。

要求：
- 必须提供患者ID或病历号
- 用于数据追溯和质量控制

请补充患者信息后重新提交。
```

**Technical Details:**
- Action: Raise `ValidationError`
- Log: Request timestamp, data source
- No inference performed

---

### MED-VALIDATION-002: Duplicate Submission

**Severity:** WARNING  
**Trigger:** Same patient data submitted within 5 minutes  
**System Behavior:** Return cached result or warn user

**User Message:**
```
ℹ️ 重复提交检测

检测到相同患者数据在短时间内重复提交。

选项：
1. 使用之前的分析结果（推荐）
2. 重新进行分析

上次分析时间：{last_analysis_time}
上次分析结果：{last_result}

是否使用缓存结果？
```

**Technical Details:**
- Check: Patient ID + data hash
- Action: Return cached result or re-run
- Log: Duplicate submission count

---

### MED-VALIDATION-003: Age-Gender Mismatch

**Severity:** WARNING  
**Trigger:** Clinical features inconsistent with demographics  
**System Behavior:** Continue with warning

**User Message:**
```
⚠️ 数据一致性警告

检测到患者信息可能存在不一致：

- 年龄：{age} 岁
- 性别：{gender}
- 异常指标：{inconsistent_features}

建议：
- 请核实患者基本信息是否正确
- 请确认临床指标数据是否录入正确

分析将继续，但请注意数据质量。
```

**Technical Details:**
- Check: Age-specific and gender-specific feature ranges
- Log: Inconsistent features
- Continue with warning flag

---

## 4. System Errors (MED-SYSTEM-xxx)

### MED-SYSTEM-001: GPU Memory Insufficient

**Severity:** ERROR  
**Trigger:** CUDA out of memory during inference  
**System Behavior:** Fallback to CPU or smaller batch

**User Message:**
```
⚠️ 系统资源不足

GPU 内存不足，系统将使用 CPU 进行分析。

影响：
- 分析时间可能延长 2-5 倍
- 分析质量不受影响

预计等待时间：{estimated_time} 秒

分析正在进行中，请稍候...
```

**Technical Details:**
- Action: Move model to CPU
- Log: GPU memory usage, batch size
- Fallback: Reduce batch size or use CPU

---

### MED-SYSTEM-002: Disk Space Low

**Severity:** WARNING  
**Trigger:** Available disk space < 1GB  
**System Behavior:** Continue but disable logging

**User Message:**
```
⚠️ 存储空间不足

系统检测到磁盘空间不足。

当前状态：
- 可用空间：{available_space} GB
- 建议空间：≥ 1 GB

影响：
- 详细日志将被禁用
- 分析功能正常

建议尽快清理磁盘空间。
```

**Technical Details:**
- Disable: Detailed logging, visualization saving
- Log: Only critical events
- Alert: System administrator

---

## 5. Fallback Strategies

### Strategy 1: Single Modality Fallback

**Trigger:** One modality missing or invalid  
**Action:**
1. Detect missing modality
2. Load single-modality model weights
3. Adjust confidence threshold
4. Provide clear user feedback

**Implementation:**
```python
if image_missing:
    model = load_tabular_only_model()
    confidence_penalty = 0.20
elif tabular_missing:
    model = load_vision_only_model()
    confidence_penalty = 0.15
```

---

### Strategy 2: Quality-Based Degradation

**Trigger:** Data quality below threshold  
**Action:**
1. Calculate quality score (0-100)
2. Apply confidence penalty proportional to quality
3. Flag result for manual review if quality < 50

**Quality Score Formula:**
```
quality_score = (
    image_quality * 0.4 +
    tabular_completeness * 0.3 +
    metadata_completeness * 0.2 +
    consistency_score * 0.1
)
```

---

### Strategy 3: Ensemble Fallback

**Trigger:** Primary model fails or times out  
**Action:**
1. Switch to lightweight backup model
2. Reduce input resolution
3. Use faster inference mode
4. Clearly indicate reduced accuracy

---

## 6. Logging and Audit

### Required Log Fields

For every inference request, log:

```json
{
  "timestamp": "2026-01-28T23:16:17Z",
  "patient_id": "P123456",
  "request_id": "req_abc123",
  "data_quality": {
    "image_available": true,
    "image_quality_score": 85,
    "tabular_completeness": 0.92,
    "overall_quality": 88
  },
  "model_info": {
    "mode": "multimodal",
    "fallback_used": false,
    "confidence": 0.87
  },
  "errors": [],
  "warnings": ["MED-DATA-005"],
  "inference_time_ms": 1234,
  "result": {
    "prediction": "class_1",
    "confidence": 0.87,
    "requires_review": false
  }
}
```

---

## 7. User Feedback Guidelines

### Tone and Language

1. **Professional but accessible**: Avoid technical jargon
2. **Action-oriented**: Always provide next steps
3. **Transparent**: Clearly state limitations
4. **Reassuring**: Emphasize safety and quality control

### Message Structure

```
[Icon] [Title]

[Problem Description]

[Impact/Details]

[Recommendations]

[Action Items]
```

### Icons

- ❌ Error (critical, cannot proceed)
- ⚠️ Warning (can proceed with caution)
- ℹ️ Info (informational, no action needed)
- ✅ Success (operation completed)

---

## 8. Configuration

### Quality Thresholds

```yaml
quality_thresholds:
  image:
    min_resolution: [128, 128]
    min_quality_score: 50
  tabular:
    min_completeness: 0.5
    max_outlier_rate: 0.2
  overall:
    min_quality_for_inference: 40
    min_quality_for_auto_report: 70

confidence_thresholds:
  high_confidence: 0.80
  medium_confidence: 0.60
  low_confidence: 0.40
  reject_threshold: 0.30

fallback_config:
  enable_single_modality: true
  enable_quality_degradation: true
  max_inference_time_seconds: 60
  retry_attempts: 2
```

---

## 9. Testing Checklist

- [ ] Test with missing image file
- [ ] Test with all-null tabular data
- [ ] Test with both modalities missing
- [ ] Test with low-resolution images
- [ ] Test with out-of-range clinical values
- [ ] Test with incomplete DICOM series
- [ ] Test with conflicting predictions
- [ ] Test with duplicate submissions
- [ ] Test GPU memory overflow
- [ ] Test disk space warning

---

## 10. Future Enhancements

1. **Adaptive Thresholds**: Learn optimal thresholds from historical data
2. **Multi-language Support**: Provide error messages in multiple languages
3. **Automated Quality Improvement**: Suggest data collection improvements
4. **Predictive Alerts**: Warn about potential issues before inference
5. **Integration with PACS**: Direct quality checks on DICOM sources

---

## Appendix A: Error Code Quick Reference

| Code | Description | Severity | Fallback |
|------|-------------|----------|----------|
| MED-DATA-001 | Missing Image | WARNING | Tabular-only |
| MED-DATA-002 | Missing Tabular | WARNING | Vision-only |
| MED-DATA-003 | Both Missing | ERROR | Abort |
| MED-DATA-004 | Low Image Quality | WARNING | Continue |
| MED-DATA-005 | Feature Out of Range | WARNING | Clip & Continue |
| MED-DATA-006 | Incomplete Series | WARNING | Use Available |
| MED-DATA-007 | Missing Metadata | INFO | Use Defaults |
| MED-MODEL-001 | Model Load Failed | ERROR | Abort |
| MED-MODEL-002 | Inference Timeout | ERROR | Abort |
| MED-MODEL-003 | Low Confidence | WARNING | Flag Review |
| MED-MODEL-004 | Conflicting Predictions | WARNING | Return All |
| MED-VALIDATION-001 | Missing Patient ID | ERROR | Abort |
| MED-VALIDATION-002 | Duplicate Submission | WARNING | Use Cache |
| MED-VALIDATION-003 | Data Inconsistency | WARNING | Continue |
| MED-SYSTEM-001 | GPU Memory Low | ERROR | Use CPU |
| MED-SYSTEM-002 | Disk Space Low | WARNING | Disable Logs |

---

**Document Maintainer:** Medical AI Engineering Team  
**Review Cycle:** Quarterly  
**Last Review:** 2026-01-28