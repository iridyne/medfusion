import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { Card } from 'antd'
import { ThunderboltOutlined } from '@ant-design/icons'

interface TrainingNodeData {
  label: string
  epochs?: number
  learningRate?: number
}

function TrainingNode({ data, selected }: NodeProps<TrainingNodeData>) {
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
            <ThunderboltOutlined />
            <span>训练</span>
          </div>
        }
      >
        <div style={{ fontSize: 12 }}>
          <div>Epochs: {data.epochs || 50}</div>
          <div>学习率: {data.learningRate || 0.001}</div>
        </div>
      </Card>
      <Handle type="source" position={Position.Right} />
    </>
  )
}

export default memo(TrainingNode)
