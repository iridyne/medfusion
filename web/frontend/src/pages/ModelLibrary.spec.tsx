import { MemoryRouter } from "react-router-dom";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import ModelLibrary from "@/pages/ModelLibrary";

describe("ModelLibrary", () => {
  it("renders the result-backend framing and handoff-friendly archive copy", () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <ModelLibrary />
      </MemoryRouter>,
    );

    expect(markup).toContain("正式版结果后台：归档、回流并展示真实训练结果");
    expect(markup).toContain("当前主线步骤：结果（3/3）");
    expect(markup).toContain("基于当前结果重开配置");
    expect(markup).toContain("基于当前结果直接重跑训练");
    expect(markup).toContain("承接训练完成后的直接深链打开");
    expect(markup).toContain("真实 run 如何回流到结果后台");
  });
});
