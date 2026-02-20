import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { Card } from 'antd'
import { DatabaseOutlined } from '@ant-design/icons'

interface DataLoaderNodeData {
  label: string
  dataPath?: string
  batchSize?: number
}

function DataLoaderNode({ data, selected }: NodeProps<DataLoaderNodeData>) {
  return (
    <Card
      size="small"
      style={{
        minWidth: 180,
        border: selected ? '2px solid #1890ff' : '1px solid #d9d9d9',
        boxShadow: selected ? '0 0 0 2px rgba(24, 144, 255, 0.2)' : undefined,
      }}
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <DatabaseOutlined />
          <span>数据加载器</span>
        </div>
      }
    >
      <div style={{ fontSize: 12 }}>
        <div>路径: {data.dataPath || '未设置'}</div>
        <div>批次大小: {data.batchSize || 32}</div>
      </div>
      <Handle type="source" position={Position.Right} />
    </Card>
  )
}

export default memo(DataLoaderNode)
