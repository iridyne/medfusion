import { MemoryRouter } from "react-router-dom";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import GettingStarted from "@/pages/GettingStarted";

describe("GettingStarted", () => {
  it("renders the formal release entry framing and primary components", () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <GettingStarted />
      </MemoryRouter>,
    );

    expect(markup).toContain("先理解正式版组件");
    expect(markup).toContain("开始问题向导");
    expect(markup).toContain("正式版组件与集成入口");
    expect(markup).toContain("问题向导");
    expect(markup).toContain("结果后台");
    expect(markup).toContain("高级模式");
    expect(markup).toContain("ComfyUI 集成");
    expect(markup).toContain("标准主线（推荐）");
    expect(markup).toContain("ComfyUI 适配线（预览）");
  });
});
