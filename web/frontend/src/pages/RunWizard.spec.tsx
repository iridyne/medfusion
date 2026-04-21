import { MemoryRouter } from "react-router-dom";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import RunWizard from "@/pages/RunWizard";

describe("RunWizard", () => {
  it("renders the question-first builder framing", () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <RunWizard />
      </MemoryRouter>,
    );

    expect(markup).toContain("先定义问题，再生成可运行的模型骨架");
    expect(markup).toContain("当前主线步骤：配置（1/3）");
    expect(markup).toContain("先说你现在要解决什么问题");
    expect(markup).toContain("官方模型模板现在来自模型数据库");
    expect(markup).toContain("当前主线默认路径");
    expect(markup).toContain("当前阶段的前台边界");
    expect(markup).toContain("问题定义");
  });
});
