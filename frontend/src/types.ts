export type Source = {
  id: string;
  label: string;
  available: boolean;
};

export type Contribution = {
  feature: string;
  description: string;
  value: number;
  weight: number;
  contribution: number;
  direction: string;
};

export type Finding = {
  text: string;
  label: string;
  probability: number;
};

export type TimelineEvent = {
  event_type: string;
  value: string;
  details: string;
  timestamp: string;
};

export type Review = {
  case_id: string;
  age: number;
  gender: string;
  scenario: string;
  score: number;
  label: boolean;
  event_count: number;
  raising: Contribution[];
  lowering: Contribution[];
  triggers: string[];
  notes: Finding[];
  timeline: TimelineEvent[];
};

export type CoverageRow = {
  trigger: string;
  meaning: string;
  cases: number;
  share: number;
  label_rate: number | null;
  lift: number | null;
};

export type TrainingReport = {
  n_train: number;
  n_test: number;
  roc_auc: number;
  average_precision: number;
  /** PR AUC of a model guessing at the base rate. ROC's equivalent is always 0.5. */
  pr_baseline: number;
  /** average_precision / pr_baseline. 1.0 means no better than chance. */
  pr_lift: number;
  recall: number;
  precision: number;
  cv_roc_auc_mean: number;
  cv_roc_auc_std: number;
  cv_average_precision_mean: number;
  cv_average_precision_std: number;
  brier: number;
  positive_rate: number;
  threshold: number;
};

export type ReviewPayload = {
  source: string;
  label_name: string;
  has_notes: boolean;
  checked: number;
  top_reviews: number;
  already_tagged_share: number;
  overall_tagged_share: number;
  tagged_found: number;
  tagged_total: number;
  reviews: Review[];
  report: TrainingReport | null;
  coverage: CoverageRow[];
  model_type: string;
};
