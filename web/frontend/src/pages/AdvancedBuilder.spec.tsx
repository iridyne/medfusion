import { MemoryRouter } from "react-router-dom";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import AdvancedBuilder from "@/pages/AdvancedBuilder";

describe("AdvancedBuilder", () => {
  it("renders the registry-backed advanced mode structure", () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <AdvancedBuilder />
      </MemoryRouter>,
    );

    expect(markup).toContain("高级建模模式");
    expect(markup).toContain("组件注册表");
    expect(markup).toContain("连接约束");
    expect(markup).toContain("当前能编译回主链的高级骨架");
    expect(markup).toContain("当前只允许停留在草稿层的组合");
  });
});
