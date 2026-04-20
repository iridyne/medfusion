# Demo And Model Creation Support Matrix

## MVP 官方支持矩阵

当前第一版 MVP 对外承诺的配置路径，只包含已经进入默认文档主链、能稳定复现结果闭环的几条。

### 官方支持配置

- `configs/public_datasets/pathmnist_quickstart.yaml`
- `configs/public_datasets/breastmnist_quickstart.yaml`
- `configs/starter/quickstart.yaml`
- `configs/starter/default.yaml`

### 使用边界

- `configs/public_datasets/breastmnist_quickstart.yaml`
  适合第一次验证公开数据主链，优先看 prepare -> train -> build-results 的闭环。
- `configs/public_datasets/pathmnist_quickstart.yaml`
  适合补第二条公开数据 quickstart，对照不同图像任务的最小结果资产。
- `configs/starter/quickstart.yaml`
  适合作为本地 YAML 主链的最短成功路径。
- `configs/starter/default.yaml`
  适合作为正式实验前的默认起点，不等价于任意模型探索模板。

### 不在本轮承诺内

- Builder 里的结构实验配置
- 仍需额外代码扩展的研究型 YAML
- 只存在于 demo 说明但没有进入主文档主链的路径

如果要判断某条配置是否进入第一版对外承诺，以这张矩阵和各目录 README 的“官方支持矩阵”说明为准。
