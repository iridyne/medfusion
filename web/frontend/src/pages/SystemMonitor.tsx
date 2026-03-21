import { useEffect, useState } from 'react'
import { Card, Row, Col, Progress } from 'antd'
import ReactECharts from 'echarts-for-react'
import { getSystemResources } from '@/api/system'

interface SystemResources {
  cpu: {
    usage_percent: number
    count: number
  }
  memory: {
    used: number
    total: number
    percent: number
  }
  gpu: Array<{
    id: number
    name: string
    memory_allocated: number
    memory_total: number
    memory_reserved: number
  }>
}

export default function SystemMonitor() {
  const [resources, setResources] = useState<SystemResources | null>(null)

  useEffect(() => {
    loadResources()
    const interval = setInterval(loadResources, 2000)
    return () => clearInterval(interval)
  }, [])

  const loadResources = async () => {
    try {
      const data = await getSystemResources()
      setResources(data)
    } catch (error) {
      console.error('加载系统资源失败:', error)
    }
  }

  const formatGB = (value: number) => {
    return `${value.toFixed(2)} GB`
  }

  return (
    <div style={{ padding: 24 }}>
      <h1>系统监控</h1>

      {resources && (
        <>
          <Row gutter={16} style={{ marginTop: 16 }}>
            <Col span={12}>
              <Card title="CPU 使用率">
                <Progress
                  type="circle"
                  percent={Math.round(resources.cpu.usage_percent)}
                  format={(percent) => `${percent}%`}
                />
                <div style={{ marginTop: 16 }}>
                  核心数: {resources.cpu.count}
                </div>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="内存使用">
                <Progress
                  type="circle"
                  percent={Math.round(resources.memory.percent)}
                  format={(percent) => `${percent}%`}
                />
                <div style={{ marginTop: 16 }}>
                  已使用: {formatGB(resources.memory.used)} / {formatGB(resources.memory.total)}
                </div>
              </Card>
            </Col>
          </Row>

          {resources.gpu.length > 0 && (
            <Card title="GPU 状态" style={{ marginTop: 16 }}>
              {resources.gpu.map((gpu) => (
                <div key={gpu.id} style={{ marginBottom: 16 }}>
                  <h4>GPU {gpu.id}: {gpu.name}</h4>
                  <Row gutter={16}>
                    <Col span={12}>
                      <div>显存使用率</div>
                      <Progress
                        percent={
                          gpu.memory_total > 0
                            ? Math.round((gpu.memory_allocated / gpu.memory_total) * 100)
                            : 0
                        }
                      />
                      <div style={{ fontSize: 12, color: '#666' }}>
                        {formatGB(gpu.memory_allocated)} / {formatGB(gpu.memory_total)}
                      </div>
                    </Col>
                    <Col span={12}>
                      <div>显存预留</div>
                      <Progress
                        percent={
                          gpu.memory_total > 0
                            ? Math.round((gpu.memory_reserved / gpu.memory_total) * 100)
                            : 0
                        }
                      />
                      <div style={{ fontSize: 12, color: '#666' }}>
                        {formatGB(gpu.memory_reserved)}
                      </div>
                    </Col>
                  </Row>
                </div>
              ))}
            </Card>
          )}
        </>
      )}
    </div>
  )
}
