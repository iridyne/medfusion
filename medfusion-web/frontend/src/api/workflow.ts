import api from './index'

export interface Workflow {
  id?: string
  name: string
  description?: string
  nodes: any[]
  edges: any[]
}

export const getNodes = async () => {
  const response = await api.get('/workflows/nodes')
  return response.data
}

export const createWorkflow = async (workflow: Workflow) => {
  const response = await api.post('/workflows/', workflow)
  return response.data
}

export const getWorkflow = async (workflowId: string) => {
  const response = await api.get(`/workflows/${workflowId}`)
  return response.data
}

export const executeWorkflow = async (workflow: Workflow) => {
  const response = await api.post('/workflows/execute', { workflow })
  return response.data
}

export const deleteWorkflow = async (workflowId: string) => {
  const response = await api.delete(`/workflows/${workflowId}`)
  return response.data
}
