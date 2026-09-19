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
  coverage: CoverageRow[];
  model_type: string;
};
