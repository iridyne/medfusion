import { useEffect, useState } from "react";
import { Card, Progress, Space } from "antd";

import { getSystemResources } from "@/api/system";
import PageScaffold from "@/components/layout/PageScaffold";

interface SystemResources {
  cpu: {
    usage_percent: number;
    count: number;
  };
  memory: {
    used: number;
    total: number;
    percent: number;
  };
  gpu: Array<{
    id: number;
    name: string;
    memory_allocated: number;
    memory_total: number;
    memory_reserved: number;
  }>;
}

export default function SystemMonitor() {
  const [resources, setResources] = useState<SystemResources | null>(null);

  useEffect(() => {
    loadResources();
    const interval = setInterval(loadResources, 2000);
    return () => clearInterval(interval);
  }, []);

  const loadResources = async () => {
    try {
      const data = await getSystemResources();
      setResources(data);
    } catch (error) {
      console.error("加载系统资源失败:", error);
    }
  };

  const formatGB = (value: number) => {
    return `${value.toFixed(2)} GB`;
  };

  return (
    <PageScaffold
      eyebrow="Resource Telemetry"
      title="观察本地工作站是否仍适合承载研究工作流"
      description="这一页把 CPU、内存和 GPU 负载集中展示成一张资源观测面，帮助你判断当前机器能否继续承担训练、推理或结果生成。"
      chips={[
        { label: "Workstation health", tone: "blue" },
        { label: "GPU visibility", tone: "rose" },
        { label: "Live polling", tone: "teal" },
      ]}
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Sampling cadence</span>
          <div className="hero-aside-panel__value">每 2 秒刷新一次资源快照</div>
          <div className="hero-aside-panel__copy">
            用来确认训练期间的 CPU、内存和 GPU 压力，不替代更细粒度的 profiler。
          </div>
          <div className="surface-note">
            {resources?.gpu.length
              ? `检测到 ${resources.gpu.length} 块 GPU。`
              : "当前未检测到 GPU，或后端尚未返回 GPU 资源。"}
          </div>
        </div>
      }
      metrics={[
        {
          label: "CPU usage",
          value: `${Math.round(resources?.cpu.usage_percent ?? 0)}%`,
          hint: `${resources?.cpu.count ?? 0} logical cores`,
          tone: "blue",
        },
        {
          label: "Memory usage",
          value: `${Math.round(resources?.memory.percent ?? 0)}%`,
          hint: resources
            ? `${formatGB(resources.memory.used)} / ${formatGB(resources.memory.total)}`
            : "等待资源快照",
          tone: "teal",
        },
        {
          label: "GPU count",
          value: `${resources?.gpu.length ?? 0}`,
          hint: "Detected accelerators",
          tone: "rose",
        },
        {
          label: "Polling",
          value: "2s",
          hint: "Resource refresh interval",
          tone: "amber",
        },
      ]}
    >
      <div className="resource-grid">
        <Card className="resource-card">
          <Space direction="vertical" size={16} style={{ width: "100%" }}>
            <div className="section-heading">
              <div>
                <div className="section-heading__eyebrow">CPU</div>
                <h2 className="section-heading__title">处理器负载</h2>
              </div>
            </div>
            <Progress
              type="circle"
              percent={Math.round(resources?.cpu.usage_percent ?? 0)}
              format={(percent) => `${percent}%`}
            />
            <div className="surface-note">
              当前工作站提供 {resources?.cpu.count ?? 0} 个逻辑核心，适合观察训练过程中的系统抖动。
            </div>
          </Space>
        </Card>

        <Card className="resource-card">
          <Space direction="vertical" size={16} style={{ width: "100%" }}>
            <div className="section-heading">
              <div>
                <div className="section-heading__eyebrow">Memory</div>
                <h2 className="section-heading__title">内存占用</h2>
              </div>
            </div>
            <Progress
              type="circle"
              percent={Math.round(resources?.memory.percent ?? 0)}
              format={(percent) => `${percent}%`}
            />
            <div className="surface-note">
              已使用 {formatGB(resources?.memory.used ?? 0)} /{" "}
              {formatGB(resources?.memory.total ?? 0)}。
            </div>
          </Space>
        </Card>
      </div>

      {resources?.gpu.length ? (
        <Card className="surface-card">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Accelerator view</div>
              <h2 className="section-heading__title">GPU 资源状态</h2>
              <p className="section-heading__description">
                同时查看显存占用和预留显存，帮助判断当前实验是否接近资源上限。
              </p>
            </div>
          </div>

          <div className="resource-grid">
            {resources.gpu.map((gpu) => (
              <div key={gpu.id} className="surface-note progress-stack">
                <div>
                  <strong>
                    GPU {gpu.id}: {gpu.name}
                  </strong>
                </div>
                <div>
                  <div>显存使用率</div>
                  <Progress
                    percent={
                      gpu.memory_total > 0
                        ? Math.round((gpu.memory_allocated / gpu.memory_total) * 100)
                        : 0
                    }
                  />
                  <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                    {formatGB(gpu.memory_allocated)} / {formatGB(gpu.memory_total)}
                  </div>
                </div>
                <div>
                  <div>显存预留</div>
                  <Progress
                    percent={
                      gpu.memory_total > 0
                        ? Math.round((gpu.memory_reserved / gpu.memory_total) * 100)
                        : 0
                    }
                  />
                  <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                    {formatGB(gpu.memory_reserved)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      ) : (
        <Card className="surface-card">
          <div className="surface-note">
            当前没有 GPU 资源快照。若预期应有 GPU，请检查后端驱动、运行环境和资源采集逻辑。
          </div>
        </Card>
      )}
    </PageScaffold>
  );
}
