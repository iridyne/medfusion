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
    expect(markup).toContain("当前主线步骤：配置适配（1/3）");
    expect(markup).toContain("打开 ComfyUI");
    expect(markup).toContain("带推荐参数进入训练");
    expect(markup).toContain("进入结果后台");
    expect(markup).toContain("带预填回到配置向导");
    expect(markup).toContain("回流到结果后台导入");
    expect(markup).toContain("适配档案");
  });
});
