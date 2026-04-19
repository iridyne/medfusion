import { MemoryRouter } from "react-router-dom";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import AdvancedBuilderCanvas from "@/pages/AdvancedBuilderCanvas";

describe("AdvancedBuilderCanvas", () => {
  it("renders the constraint-aware node prototype shell", () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <AdvancedBuilderCanvas />
      </MemoryRouter>,
    );

    expect(markup).toContain("高级模式节点图原型");
    expect(markup).toContain("编译边界检查");
    expect(markup).toContain("图编译结果");
    expect(markup).toContain("当前原型不做什么");
  });
});
