import { MemoryRouter } from "react-router-dom";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import TrainingMonitor from "@/pages/TrainingMonitor";

describe("TrainingMonitor", () => {
  it("renders result handoff framing for completed training runs", () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <TrainingMonitor />
      </MemoryRouter>,
    );

    expect(markup).toContain("Artifact handoff");
    expect(markup).toContain("训练完成后，结果输出页签会给出直接进入结果后台的入口");
    expect(markup).toContain("结果输出");
    expect(markup).toContain("主线快捷跳转：配置 -&gt; 训练 -&gt; 结果");
    expect(markup).toContain("基于当前任务重开配置");
    expect(markup).toContain("回到配置主线");
  });
});
