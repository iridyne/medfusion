import DataLoaderNode from './DataLoaderNode'
import ModelNode from './ModelNode'
import TrainingNode from './TrainingNode'
import EvaluationNode from './EvaluationNode'

export const nodeTypes = {
  dataLoader: DataLoaderNode,
  model: ModelNode,
  training: TrainingNode,
  evaluation: EvaluationNode,
}

export { DataLoaderNode, ModelNode, TrainingNode, EvaluationNode }
