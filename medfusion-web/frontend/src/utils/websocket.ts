/**
 * WebSocket 客户端，支持自动重连
 */

type MessageHandler = (data: any) => void
type ErrorHandler = (error: Event) => void
type ConnectionHandler = () => void

interface WebSocketClientOptions {
  url: string
  maxReconnectAttempts?: number
  reconnectInterval?: number
  heartbeatInterval?: number
  onMessage?: MessageHandler
  onError?: ErrorHandler
  onOpen?: ConnectionHandler
  onClose?: ConnectionHandler
}

class WebSocketClient {
  private ws: WebSocket | null = null
  private url: string
  private reconnectAttempts = 0
  private maxReconnectAttempts: number
  private reconnectInterval: number
  private heartbeatInterval: number
  private heartbeatTimer: NodeJS.Timeout | null = null
  private reconnectTimer: NodeJS.Timeout | null = null
  private isManualClose = false
  
  private onMessageHandler?: MessageHandler
  private onErrorHandler?: ErrorHandler
  private onOpenHandler?: ConnectionHandler
  private onCloseHandler?: ConnectionHandler

  constructor(options: WebSocketClientOptions) {
    this.url = options.url
    this.maxReconnectAttempts = options.maxReconnectAttempts ?? 5
    this.reconnectInterval = options.reconnectInterval ?? 3000
    this.heartbeatInterval = options.heartbeatInterval ?? 30000
    
    this.onMessageHandler = options.onMessage
    this.onErrorHandler = options.onError
    this.onOpenHandler = options.onOpen
    this.onCloseHandler = options.onClose
  }

  /**
   * 连接 WebSocket
   */
  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected')
      return
    }

    try {
      this.ws = new WebSocket(this.url)
      
      this.ws.onopen = () => {
        console.log('WebSocket connected')
        this.reconnectAttempts = 0
        this.startHeartbeat()
        this.onOpenHandler?.()
      }

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          this.onMessageHandler?.(data)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        this.onErrorHandler?.(error)
      }

      this.ws.onclose = () => {
        console.log('WebSocket closed')
        this.stopHeartbeat()
        this.onCloseHandler?.()
        
        // 如果不是手动关闭，尝试重连
        if (!this.isManualClose) {
          this.reconnect()
        }
      }
    } catch (error) {
      console.error('Failed to create WebSocket:', error)
      this.reconnect()
    }
  }

  /**
   * 重连
   */
  private reconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1)
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
    
    this.reconnectTimer = setTimeout(() => {
      this.connect()
    }, delay)
  }

  /**
   * 发送消息
   */
  send(data: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const message = typeof data === 'string' ? data : JSON.stringify(data)
      this.ws.send(message)
    } else {
      console.warn('WebSocket is not connected')
    }
  }

  /**
   * 关闭连接
   */
  close(): void {
    this.isManualClose = true
    this.stopHeartbeat()
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  /**
   * 启动心跳
   */
  private startHeartbeat(): void {
    this.stopHeartbeat()
    
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' })
      }
    }, this.heartbeatInterval)
  }

  /**
   * 停止心跳
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  /**
   * 获取连接状态
   */
  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED
  }

  /**
   * 是否已连接
   */
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

export default WebSocketClient
