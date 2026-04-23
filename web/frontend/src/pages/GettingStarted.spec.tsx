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

    expect(markup).toContain("开始一次研究运行");
    expect(markup).toContain("从数据检查开始");
    expect(markup).toContain("直接进入问题向导");
    expect(markup).toContain("推荐主线");
    expect(markup).toContain("模型搭建入口");
    expect(markup).toContain("官方模型库");
    expect(markup).toContain("本机自定义模型");
    expect(markup).toContain("ComfyUI 适配入口");
    expect(markup).toContain("常用跳转");
    expect(markup).toContain("启动推荐训练");
    expect(markup).toContain("直接查看结果后台");
  });
});
