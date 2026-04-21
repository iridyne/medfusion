export type AdvancedBuilderFamily =
  | "data_input"
  | "vision_backbone"
  | "tabular_encoder"
  | "fusion"
  | "head"
  | "training_strategy";

export type AdvancedBuilderStatus =
  | "compile_ready"
  | "conditional"
  | "draft_only";

export interface AdvancedBuilderComponent {
  id: string;
  family: AdvancedBuilderFamily;
  label: string;
  status: AdvancedBuilderStatus;
  description: string;
  schemaPath?: string;
  inputs?: string[];
  outputs?: string[];
  notes?: string[];
  advancedBuilderContract?: {
    patch_target_hints?: Array<{
      path: string;
      mode: string;
      description: string;
    }>;
  };
}

export interface AdvancedBuilderConnectionRule {
  fromFamily: AdvancedBuilderFamily;
  toFamily: AdvancedBuilderFamily;
  status: "required" | "conditional" | "blocked";
  description: string;
}

export interface AdvancedBuilderBlueprint {
  id: string;
  label: string;
  status: "compile_ready" | "draft_only";
  description: string;
  components: string[];
  compilesTo?: string;
  blockers?: string[];
  recommendedPreset?: string;
}
