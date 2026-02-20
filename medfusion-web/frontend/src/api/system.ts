import api from './index'

export const getSystemInfo = async () => {
  const response = await api.get('/system/info')
  return response.data
}

export const getSystemResources = async () => {
  const response = await api.get('/system/resources')
  return response.data
}
