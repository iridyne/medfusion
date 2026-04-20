import { describe, expect, it } from "vitest";

describe("start experience contract", () => {
  it("defines the formal release entry payload", async () => {
    const startExperience = await import("@/config/startExperience").catch(
      () => null,
    );

    expect(startExperience?.DEFAULT_START_ROUTE).toBe("/start");
    expect(startExperience?.START_COMPONENTS).toHaveLength(5);
    expect(
      startExperience?.START_COMPONENTS?.some(
        (item) => item.route === "/config/comfyui",
      ),
    ).toBe(true);
    expect(startExperience?.START_PRIMARY_FLOW).toEqual([
      "组件介绍",
      "问题定义",
      "骨架推荐",
      "训练执行",
      "结果后台",
    ]);
    expect(startExperience?.START_RECOMMENDED_WORKFLOW).toHaveLength(4);
    expect(startExperience?.START_COMFYUI_OPTIONAL_MODULE).toHaveLength(3);
    expect(startExperience?.START_MODE_POSITIONING?.advancedMode).toContain(
      "高级模式",
    );
  });
});
