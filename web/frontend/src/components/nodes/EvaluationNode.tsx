import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { Card } from 'antd'
import { LineChartOutlined } from '@ant-design/icons'

interface EvaluationNodeData {
  label: string
  metrics?: string[]
}

function EvaluationNode({ data, selected }: NodeProps<EvaluationNodeData>) {
  return (
    <>
      <Handle type="target" position={Position.Left} />
      <Card
        size="small"
        style={{
          minWidth: 180,
          border: selected ? '2px solid #1890ff' : '1px solid #d9d9d9',
          boxShadow: selected ? '0 0 0 2px rgba(24, 144, 255, 0.2)' : undefined,
        }}
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <LineChartOutlined />
            <span>评估</span>
          </div>
        }
      >
        <div style={{ fontSize: 12 }}>
          <div>指标: {data.metrics?.join(', ') || 'accuracy, loss'}</div>
        </div>
      </Card>
    </>
  )
}

export default memo(EvaluationNode)
