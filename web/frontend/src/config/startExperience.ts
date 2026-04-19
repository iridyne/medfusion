export const DEFAULT_START_ROUTE = "/start";

export interface StartComponentItem {
  key: string;
  title: string;
  description: string;
  route: string;
  accent: "amber" | "blue" | "teal" | "rose";
}

export const START_COMPONENTS: StartComponentItem[] = [
  {
    key: "problem-wizard",
    title: "问题向导",
    description: "先从问题定义出发，再映射到当前 runtime 真正支持的模型骨架。",
    route: "/config",
    accent: "amber",
  },
  {
    key: "training-monitor",
    title: "训练执行",
    description: "继续使用真实训练链路，Web 负责提交、观察和解释状态，而不是发明第二套执行语义。",
    route: "/training",
    accent: "rose",
  },
  {
    key: "result-backend",
    title: "结果后台",
    description: "把 summary、validation、report 和可视化 artifact 组织成结构化结果资产。",
    route: "/models",
    accent: "teal",
  },
  {
    key: "workbench",
    title: "工作台总览",
    description: "在第一次引导之后回到总览页，继续看最近运行、下一步动作和资产状态。",
    route: "/workbench",
    accent: "blue",
  },
];

export const START_PRIMARY_FLOW = [
  "组件介绍",
  "问题定义",
  "骨架推荐",
  "训练执行",
  "结果后台",
] as const;

export const START_MODE_POSITIONING = {
  defaultMode: "问题向导 -> 模板骨架 -> 参数编辑",
  advancedMode: "节点式编辑保留为高级模式，不作为默认首页",
} as const;
