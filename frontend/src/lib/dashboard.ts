export interface RegisteredModelResponse {
  model_name: string;
  version: number;
  artifact_path: string;
  metrics: Record<string, number>;
  created_at: string;
  status: string;
}

export interface ModelRegistryResponse {
  models: RegisteredModelResponse[];
  count: number;
}

export interface ExperimentResponse {
  experiment_id: string;
  timestamp: string;
  model_name: string;
  dataset_version: string;
  hyperparameters: Record<string, number | string>;
  metrics: Record<string, number>;
  training_time_seconds: number;
  model_version: number;
}

export interface ExperimentHistoryResponse {
  experiments: ExperimentResponse[];
  count: number;
}

export interface MonitoringRecordResponse {
  timestamp: string;
  model_name: string;
  prediction_count: number;
  mean_prediction: number;
  std_prediction: number;
  min_prediction: number;
  max_prediction: number;
  missing_inputs: number;
  out_of_range_inputs: number;
}

export interface MonitoringResponse {
  records: MonitoringRecordResponse[];
  count: number;
}

export interface ModelInfoResponse {
  current_model: string;
  version: number;
  created_at?: string;
  available_models: string[];
  models: Array<{
    model_name: string;
    version: number;
    artifact_path: string;
    status: string;
    created_at?: string;
    metrics?: Record<string, number>;
    dataset_size?: number | null;
    training_samples?: number | null;
    testing_samples?: number | null;
    feature_names?: string[];
    feature_count?: number | null;
    target_column?: string | null;
    feature_types?: Record<string, string>;
  }>;
  count: number;
}

export interface HealthResponse {
  status: string;
  model_loaded?: boolean;
}

export interface PredictionHealthResponse {
  status: string;
  model_loaded?: boolean;
}

export interface PredictionResponse {
  prediction: number;
  model_name: string;
  model_version: number;
  processing_time_ms: number;
}

export async function fetchJson<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const url = path.startsWith("http") ? path : path;

  const response = await fetch(url, init);

  if (!response.ok) {
    let message = response.statusText || "Request failed.";
    const rawBody = await response.text();

    if (rawBody) {
      try {
        const body = JSON.parse(rawBody);
        if (typeof body === "string") {
          message = body;
        } else if (typeof body?.detail === "string") {
          message = body.detail;
        } else if (typeof body?.message === "string") {
          message = body.message;
        }
      } catch {
        message = rawBody;
      }
    }

    throw new Error(message);
  }

  return response.json();
}
