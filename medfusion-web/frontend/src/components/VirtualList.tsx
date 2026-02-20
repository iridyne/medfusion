import React from 'react'
import { FixedSizeList, ListChildComponentProps, VariableSizeList } from 'react-window'
import AutoSizer from 'react-virtualized-auto-sizer'
import { Empty, Spin } from 'antd'

interface VirtualListProps<T> {
  data: T[]
  itemHeight?: number | ((index: number) => number)
  renderItem: (item: T, index: number) => React.ReactNode
  loading?: boolean
  emptyText?: string
  className?: string
  style?: React.CSSProperties
  overscanCount?: number
}

/**
 * 虚拟滚动列表组件
 *
 * 用于渲染大量数据列表，只渲染可见区域的项目，提高性能
 *
 * @example
 * ```tsx
 * <VirtualList
 *   data={models}
 *   itemHeight={100}
 *   renderItem={(model, index) => (
 *     <ModelCard key={model.id} model={model} />
 *   )}
 * />
 * ```
 */
function VirtualList<T>({
  data,
  itemHeight = 80,
  renderItem,
  loading = false,
  emptyText = '暂无数据',
  className,
  style,
  overscanCount = 5,
}: VirtualListProps<T>) {
  // 如果正在加载，显示加载状态
  if (loading) {
    return (
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: 400,
          ...style,
        }}
        className={className}
      >
        <Spin size="large" tip="加载中..." />
      </div>
    )
  }

  // 如果没有数据，显示空状态
  if (!data || data.length === 0) {
    return (
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: 400,
          ...style,
        }}
        className={className}
      >
        <Empty description={emptyText} />
      </div>
    )
  }

  // 判断是固定高度还是动态高度
  const isFixedHeight = typeof itemHeight === 'number'

  // 渲染项目的包装函数
  const Row = ({ index, style: itemStyle }: ListChildComponentProps) => {
    const item = data[index]
    return (
      <div style={itemStyle}>
        {renderItem(item, index)}
      </div>
    )
  }

  // 获取项目高度（用于动态高度）
  const getItemSize = (index: number): number => {
    if (typeof itemHeight === 'function') {
      return itemHeight(index)
    }
    return itemHeight
  }

  return (
    <div style={{ height: '100%', width: '100%', ...style }} className={className}>
      <AutoSizer>
        {({ height, width }) => {
          if (isFixedHeight) {
            // 固定高度列表
            return (
              <FixedSizeList
                height={height}
                width={width}
                itemCount={data.length}
                itemSize={itemHeight as number}
                overscanCount={overscanCount}
              >
                {Row}
              </FixedSizeList>
            )
          } else {
            // 动态高度列表
            return (
              <VariableSizeList
                height={height}
                width={width}
                itemCount={data.length}
                itemSize={getItemSize}
                overscanCount={overscanCount}
              >
                {Row}
              </VariableSizeList>
            )
          }
        }}
      </AutoSizer>
    </div>
  )
}

export default VirtualList
