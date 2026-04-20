import { MemoryRouter } from "react-router-dom";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import ComfyUIBridge from "@/pages/ComfyUIBridge";

describe("ComfyUIBridge", () => {
  it("renders the comfyui bridge framing", () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <ComfyUIBridge />
      </MemoryRouter>,
    );

    expect(markup).toContain("ComfyUI 上线入口");
    expect(markup).toContain("连接配置");
    expect(markup).toContain("最小上线链路");
    expect(markup).toContain("打开 ComfyUI");
  });
});
