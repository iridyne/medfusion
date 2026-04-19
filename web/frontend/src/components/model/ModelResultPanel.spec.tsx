import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { Model } from "@/api/models";
import ModelResultPanel from "@/components/model/ModelResultPanel";

const mockModel = {
  id: 1,
  name: "demo-run",
  backbone: "resnet18",
  num_classes: 2,
  model_path: "outputs/demo-run/checkpoints/best.pth",
  checkpoint_path: "outputs/demo-run/checkpoints/best.pth",
  config_path: "configs/starter/quickstart.yaml",
  created_at: "2026-04-19T00:00:00Z",
  accuracy: 0.91,
  training_time: 95,
  dataset_name: "BreastMNIST",
  tags: ["oss", "demo"],
  config: {
    import_source: "web",
    source_context: {
      source_type: "advanced_builder",
      entrypoint: "advanced-builder-canvas",
      blueprint_id: "quickstart_multimodal",
    },
    result_summary: {
      experiment_name: "demo-run",
      split: "test",
      best_accuracy: 0.91,
    },
  },
  validation: {
    overview: {
      sample_count: 32,
      positive_class_label: "positive",
      macro_f1: 0.88,
      balanced_accuracy: 0.9,
      auc: 0.93,
    },
    dataset: {
      name: "BreastMNIST",
      class_distribution: [
        { label: "negative", count: 16, rate: 0.5 },
        { label: "positive", count: 16, rate: 0.5 },
      ],
    },
  },
  result_files: [
    {
      key: "summary",
      label: "结果摘要",
      path: "outputs/demo-run/reports/summary.json",
      exists: true,
      is_image: false,
    },
  ],
} as Model;

describe("ModelResultPanel", () => {
  it("renders the formal result backend four-layer structure", () => {
    const markup = renderToStaticMarkup(<ModelResultPanel model={mockModel} />);

    expect(markup).toContain("1. 结论层");
    expect(markup).toContain("2. 指标层");
    expect(markup).toContain("3. 可视化层");
    expect(markup).toContain("4. 文件层");
    expect(markup).toContain("结果交付摘要");
    expect(markup).toContain("结果文件");
    expect(markup).toContain("advanced_builder");
    expect(markup).toContain("advanced-builder-canvas");
    expect(markup).toContain("quickstart_multimodal");
  });
});
