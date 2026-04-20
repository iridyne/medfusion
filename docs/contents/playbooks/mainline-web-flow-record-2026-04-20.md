# Web 主链实录（2026-04-20）

> 文档状态：Stable
>
> 对应模板：`docs/contents/playbooks/mainline-web-flow-record.md`

## 1. 执行环境

- 日期：2026-04-20
- OS：Windows（开发机）
- Repo：`C:\Users\Administrator\Projects\medfusion`
- 证据目录：`outputs/evidence/mainline-web-flow/2026-04-20/`

## 2. Web 主链入口可达性

启动命令：

```powershell
uv run medfusion start --host 127.0.0.1 --port 18081 --no-browser
```

路由检查结果（见 `web-route-check.txt`）：

- `GET /health` -> `200`
- `GET /start` -> `200`
- `GET /config` -> `200`
- `GET /training` -> `200`
- `GET /models` -> `200`

## 3. 配置与训练闭环

执行命令：

```powershell
uv run medfusion validate-config --config configs/starter/quickstart.yaml
uv run medfusion train --config configs/starter/quickstart.yaml
uv run medfusion build-results --config configs/starter/quickstart.yaml --checkpoint outputs/quickstart/checkpoints/best.pth
```

结果：

- `validate-config` 成功（`errors=0`，有 1 条 warning）
- `train` 成功（3 个 epoch，best checkpoint 产出）
- `build-results` 成功（metrics/summary/report/artifacts 产出）

## 4. 本轮发现

首次执行 `build-results` 使用了错误的 checkpoint 路径：

```powershell
outputs/starter/quickstart/checkpoints/best.pth
```

导致 `FileNotFoundError`。修正为：

```powershell
outputs/quickstart/checkpoints/best.pth
```

后再次执行成功（见 `build-results-fixed.log`）。

## 5. 产物核验

关键产物已确认存在（见 `artifact-check.txt`）：

- `outputs/quickstart/checkpoints/best.pth`
- `outputs/quickstart/logs/history.json`
- `outputs/quickstart/metrics/metrics.json`
- `outputs/quickstart/metrics/validation.json`
- `outputs/quickstart/reports/summary.json`
- `outputs/quickstart/reports/report.md`
- `outputs/quickstart/artifacts/visualizations/roc_curve.png`

## 6. 结论

`/start -> /config -> /training -> /models` 主链在本机已可复现并形成证据闭环。  
同日已补齐 Windows `smoke + uninstall` 发布证据。

## 7. 同日补充（release smoke）

执行：

```powershell
uv run python scripts/release_smoke.py --mode local
```

结果：通过（`Formal-release smoke completed.`）。  
日志：`outputs/evidence/mainline-web-flow/2026-04-20/release-smoke-local.log`

## 8. 同日补充（uninstall 命令可用性）

执行（dry-run）：

```powershell
uv run medfusion uninstall --dry-run --yes --json
uv run medfusion uninstall --purge-data --dry-run --yes --json
```

结果：通过（命令可用，目标路径解析符合预期）。  
日志：

- `outputs/evidence/mainline-web-flow/2026-04-20/uninstall-dry-run.json`
- `outputs/evidence/mainline-web-flow/2026-04-20/uninstall-purge-dry-run.json`

隔离沙箱实跑（非 dry-run）：

- `outputs/evidence/mainline-web-flow/2026-04-20/uninstall-sandbox-run.json`
- `outputs/evidence/mainline-web-flow/2026-04-20/uninstall-sandbox-postcheck.json`

真实用户目录场景实跑（非 dry-run）：

- `outputs/evidence/mainline-web-flow/2026-04-20/uninstall-userdir-sim-run.json`
- `outputs/evidence/mainline-web-flow/2026-04-20/uninstall-userdir-sim-postcheck.json`
