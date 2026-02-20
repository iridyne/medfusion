import { Card, Space } from 'antd'
import {
  DatabaseOutlined,
  ExperimentOutlined,
  ThunderboltOutlined,
  LineChartOutlined,
} from '@ant-design/icons'

interface NodePaletteProps {
  onAddNode: (type: string) => void
}

const nodeDefinitions = [
  {
    type: 'dataLoader',
    label: '数据加载器',
    icon: <DatabaseOutlined />,
    color: '#1890ff',
  },
  {
    type: 'model',
    label: '模型',
    icon: <ExperimentOutlined />,
    color: '#52c41a',
  },
  {
    type: 'training',
    label: '训练',
    icon: <ThunderboltOutlined />,
    color: '#faad14',
  },
  {
    type: 'evaluation',
    label: '评估',
    icon: <LineChartOutlined />,
    color: '#722ed1',
  },
]

export default function NodePalette({ onAddNode }: NodePaletteProps) {
  return (
    <Card
      title="节点库"
      size="small"
      style={{
        position: 'absolute',
        top: 16,
        left: 16,
        zIndex: 10,
        width: 200,
      }}
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        {nodeDefinitions.map((node) => (
          <Card
            key={node.type}
            size="small"
            hoverable
            onClick={() => onAddNode(node.type)}
            style={{
              cursor: 'pointer',
              borderLeft: `4px solid ${node.color}`,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ color: node.color, fontSize: 18 }}>
                {node.icon}
              </span>
              <span>{node.label}</span>
            </div>
          </Card>
        ))}
      </Space>
    </Card>
  )
}
