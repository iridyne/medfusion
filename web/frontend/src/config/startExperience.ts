export const DEFAULT_START_ROUTE = "/start";

export interface StartCheckItem {
  key: string;
  title: string;
  description: string;
}

export const START_CHECKS: StartCheckItem[] = [
  {
    key: "python-env",
    title: "Python 与依赖",
    description: "确认当前环境已经完成 uv sync，并且 medfusion 命令可调用。",
  },
  {
    key: "frontend-assets",
    title: "Web 资源",
    description: "确认前端静态资源可用，避免首次启动后只看到空壳页面。",
  },
  {
    key: "writable-output",
    title: "输出目录",
    description: "确认 outputs 和数据目录可写，避免训练完成后产物无法落盘。",
  },
  {
    key: "recommended-profile",
    title: "推荐 quickstart",
    description: "默认使用公开数据 quickstart 路径完成第一次成功闭环。",
  },
];

export const START_NEXT_STEPS = [
  "quickstart",
  "bring-your-own-yaml",
] as const;
