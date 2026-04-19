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
    expect(markup).toContain("先说你现在要解决什么问题");
    expect(markup).toContain("第一次先跑通一条真实主链");
    expect(markup).toContain("当前阶段的前台边界");
    expect(markup).toContain("问题定义");
  });
});
