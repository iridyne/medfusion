import React, { useRef, useEffect, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { EChartsOption } from 'echarts'
import { Skeleton } from 'antd'

interface LazyChartProps {
  option: EChartsOption
  style?: React.CSSProperties
  className?: string
  loading?: boolean
  notMerge?: boolean
  lazyUpdate?: boolean
  theme?: string
  opts?: {
    renderer?: 'canvas' | 'svg'
    width?: number | string
    height?: number | string
  }
  onChartReady?: (chart: any) => void
  onEvents?: Record<string, Function>
  showLoading?: boolean
  loadingOption?: any
  rootMargin?: string
  threshold?: number
}

/**
 * 懒加载图表组件
 *
 * 使用 Intersection Observer API 实现图表懒加载
 * 只有当图表进入视口时才渲染，提高页面性能
 *
 * @example
 * ```tsx
 * <LazyChart
 *   option={chartOption}
 *   style={{ height: 400 }}
 *   rootMargin="100px"
 * />
 * ```
 */
const LazyChart: React.FC<LazyChartProps> = ({
  option,
  style,
  className,
  loading = false,
  notMerge = false,
  lazyUpdate = false,
  theme,
  opts,
  onChartReady,
  onEvents,
  showLoading = false,
  loadingOption,
  rootMargin = '100px',
  threshold = 0.1,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isVisible, setIsVisible] = useState(false)
  const [hasRendered, setHasRendered] = useState(false)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // 创建 Intersection Observer
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !hasRendered) {
            // 图表进入视口
            setIsVisible(true)
            setHasRendered(true)
            // 一旦渲染过就不再观察
            observer.unobserve(container)
          }
        })
      },
      {
        rootMargin,
        threshold,
      }
    )

    observer.observe(container)

    return () => {
      observer.disconnect()
    }
  }, [hasRendered, rootMargin, threshold])

  // 默认样式
  const defaultStyle: React.CSSProperties = {
    width: '100%',
    height: 400,
    ...style,
  }

  // 如果正在加载，显示骨架屏
  if (loading) {
    return (
      <div ref={containerRef} style={defaultStyle} className={className}>
        <Skeleton.Node active style={{ width: '100%', height: '100%' }}>
          <div style={{ width: '100%', height: '100%' }} />
        </Skeleton.Node>
      </div>
    )
  }

  // 如果还未进入视口，显示占位符
  if (!isVisible) {
    return (
      <div
        ref={containerRef}
        style={{
          ...defaultStyle,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: '#f5f5f5',
          border: '1px dashed #d9d9d9',
          borderRadius: 4,
        }}
        className={className}
      >
        <span style={{ color: '#999' }}>图表加载中...</span>
      </div>
    )
  }

  // 渲染图表
  return (
    <div ref={containerRef} style={defaultStyle} className={className}>
      <ReactECharts
        option={option}
        style={{ width: '100%', height: '100%' }}
        notMerge={notMerge}
        lazyUpdate={lazyUpdate}
        theme={theme}
        opts={opts}
        onChartReady={onChartReady}
        onEvents={onEvents}
        showLoading={showLoading}
        loadingOption={loadingOption}
      />
    </div>
  )
}

export default LazyChart
