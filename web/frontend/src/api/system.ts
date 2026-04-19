import api from './index'

export const getSystemFeatures = async () => {
  const response = await api.get('/system/features')
  return response.data
}

export const getSystemInfo = async () => {
  const response = await api.get('/system/info')
  return response.data
}

export const getSystemResources = async () => {
  const response = await api.get('/system/resources')
  return response.data
}
