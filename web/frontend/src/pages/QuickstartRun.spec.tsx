import { MemoryRouter } from "react-router-dom";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import QuickstartRun from "@/pages/QuickstartRun";

describe("QuickstartRun", () => {
  it("renders the guided quickstart stages and result contract", () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <QuickstartRun />
      </MemoryRouter>,
    );

    expect(markup).toContain("第一次运行链路");
    expect(markup).toContain("准备公开数据");
    expect(markup).toContain("校验配置");
    expect(markup).toContain("启动训练");
    expect(markup).toContain("构建结果");
    expect(markup).toContain("summary.json");
    expect(markup).toContain("YAML + CLI");
  });
});
