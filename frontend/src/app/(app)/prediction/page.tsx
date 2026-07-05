"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, Loader2, Sparkles, TrendingUp } from "lucide-react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { fetchJson, type ModelInfoResponse, type PredictionResponse } from "@/lib/dashboard";

interface FeatureField {
  name: string;
  label: string;
  type: "number" | "categorical" | "boolean";
  options?: string[];
}

interface PredictionFormState {
  values: Record<string, string>;
  errors: Record<string, string>;
}

interface ExplainabilityPayload {
  confidence_score?: number;
  ranked_features?: Array<{ feature?: string; contribution?: number }>;
}

function toLabel(name: string) {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function normalizeFeatureType(type?: string) {
  const value = type?.toLowerCase();
  if (value === "categorical") return "categorical";
  if (value === "boolean") return "boolean";
  return "number";
}

export default function PredictionPage() {
  const router = useRouter();
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loadingModel, setLoadingModel] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predictionTimestamp, setPredictionTimestamp] = useState<string | null>(null);
  const [explainability, setExplainability] = useState<ExplainabilityPayload | null>(null);
  const [loadingExplainability, setLoadingExplainability] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formState, setFormState] = useState<PredictionFormState>({ values: {}, errors: {} });

  const productionModel = useMemo(() => {
    if (!modelInfo?.models?.length) return null;
    const xgbProductionModel = modelInfo.models.find(
      (entry) => entry.status === "Production" && entry.model_name === "XGBRegressor"
    );
    return xgbProductionModel ?? modelInfo.models.find((entry) => entry.status === "Production") ?? modelInfo.models[0];
  }, [modelInfo]);

  const featureFields = useMemo<FeatureField[]>(() => {
    if (!productionModel?.feature_names?.length) {
      return [];
    }

    return productionModel.feature_names.map((name) => {
      const featureType = normalizeFeatureType(productionModel.feature_types?.[name]);
      const options = featureType === "categorical" ? ["yes", "no"] : undefined;
      return {
        name,
        label: toLabel(name),
        type: featureType,
        options,
      };
    });
  }, [productionModel]);

  const modelMetrics = useMemo(() => {
    if (!productionModel?.metrics) return [];
    return Object.entries(productionModel.metrics).slice(0, 4);
  }, [productionModel]);

  const loadModelInfo = useCallback(async () => {
    try {
      setLoadingModel(true);
      const payload = await fetchJson<ModelInfoResponse>("/api/v1/models");
      setModelInfo(payload);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load model metadata.");
    } finally {
      setLoadingModel(false);
    }
  }, []);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      void loadModelInfo();
    }, 0);

    return () => window.clearTimeout(timeoutId);
  }, [loadModelInfo]);

  const formValues = useMemo(() => {
    if (!featureFields.length) {
      return formState.values;
    }

    return Object.fromEntries(featureFields.map((field) => [field.name, formState.values[field.name] ?? ""]));
  }, [featureFields, formState.values]);

  const validateFields = useCallback(() => {
    const nextErrors: Record<string, string> = {};

    for (const field of featureFields) {
      const rawValue = formState.values[field.name]?.trim() ?? "";
      if (!rawValue) {
        nextErrors[field.name] = `${field.label} is required.`;
        continue;
      }

      if (field.type === "number") {
        const numericValue = Number(rawValue);
        if (!Number.isFinite(numericValue)) {
          nextErrors[field.name] = `${field.label} must be a valid number.`;
        } else if (Math.abs(numericValue) > 1_000_000_000_000) {
          nextErrors[field.name] = `${field.label} is too large.`;
        }
      }
    }

    return nextErrors;
  }, [featureFields, formState.values]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setPrediction(null);
    setPredictionTimestamp(null);
    setExplainability(null);
    setError(null);

    const nextErrors = validateFields();
    if (Object.keys(nextErrors).length > 0) {
      setFormState((current) => ({ ...current, errors: nextErrors }));
      return;
    }

    try {
      setSubmitting(true);
      const payload = {
        features: Object.fromEntries(
          Object.entries(formState.values).map(([key, value]) => {
            const field = featureFields.find((entry) => entry.name === key);
            if (field?.type === "number") {
              return [key, Number(value)];
            }
            if (field?.type === "boolean") {
              return [key, value === "true" || value === "1" || value === "yes"];
            }
            return [key, value];
          })
        ),
      };

      const response = await fetchJson<PredictionResponse>("/api/v1/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      setPrediction(response);
      setPredictionTimestamp(new Date().toLocaleString());

      try {
        setLoadingExplainability(true);
        const explainabilityPayload = await fetchJson<ExplainabilityPayload>("/api/v1/explainability/latest");
        setExplainability(explainabilityPayload);
      } catch {
        setExplainability(null);
      } finally {
        setLoadingExplainability(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction request failed.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleChange = (name: string, value: string) => {
    setFormState((current) => ({
      ...current,
      values: { ...current.values, [name]: value },
      errors: { ...current.errors, [name]: "" },
    }));
  };

  const handleToggleBoolean = (name: string, checked: boolean) => {
    handleChange(name, checked ? "true" : "false");
  };

  return (
    <div className="space-y-6">
      <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-6 shadow-2xl shadow-black/20 backdrop-blur-xl">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.3em] text-sky-400">Predict ETA</p>
            <h1 className="mt-2 text-2xl font-semibold text-white">Production prediction workspace</h1>
            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              Submit delivery features and receive a live ETA prediction from the current production model.
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-300">
            <p className="font-medium text-white">Production model</p>
            <p className="mt-1 text-sky-400">{loadingModel ? "Loading..." : productionModel?.model_name ?? "Unavailable"}</p>
            <p className="text-xs text-slate-500">Version {loadingModel ? "—" : productionModel?.version ?? "—"}</p>
            <p className="text-xs text-slate-500">Status {loadingModel ? "—" : productionModel?.status ?? "Unknown"}</p>
          </div>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.35fr_0.95fr]">
        <Card>
          <div className="flex items-center gap-3">
            <div className="rounded-2xl bg-sky-500/15 p-2 text-sky-400">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">ETA request</h2>
              <p className="text-sm text-slate-400">Dynamic feature form from the active production model metadata.</p>
            </div>
          </div>

          <div className="mt-4 grid gap-3 rounded-2xl border border-white/10 bg-white/5 p-3 text-sm text-slate-300 sm:grid-cols-3">
            <div>
              <p className="text-slate-500">Model name</p>
              <p className="font-medium text-white">{productionModel?.model_name ?? "—"}</p>
            </div>
            <div>
              <p className="text-slate-500">Feature count</p>
              <p className="font-medium text-white">{productionModel?.feature_count ?? featureFields.length}</p>
            </div>
            <div>
              <p className="text-slate-500">Training date</p>
              <p className="font-medium text-white">{productionModel?.created_at ? new Date(productionModel.created_at).toLocaleDateString() : "—"}</p>
            </div>
          </div>

          <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 p-3 text-sm text-slate-300">
            <p className="text-slate-500">Feature names</p>
            <p className="mt-1 font-medium text-white">{productionModel?.feature_names?.join(", ") || "—"}</p>
            {modelMetrics.length ? (
              <div className="mt-3 space-y-1">
                <p className="text-slate-500">Metrics</p>
                {modelMetrics.map(([key, value]) => (
                  <p key={key} className="text-white">
                    <span className="text-slate-400">{key}:</span> {typeof value === "number" ? value.toFixed(3) : String(value)}
                  </p>
                ))}
              </div>
            ) : null}
          </div>

          <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
            {featureFields.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2">
                {featureFields.map((field) => (
                  <label key={field.name} className="space-y-2 text-sm text-slate-300">
                    <span className="font-medium">{field.label}</span>
                    {field.type === "boolean" ? (
                      <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-3 py-2">
                        <input
                          id={field.name}
                          aria-label={field.label}
                          type="checkbox"
                          checked={formValues[field.name] === "true"}
                          onChange={(event) => handleToggleBoolean(field.name, event.target.checked)}
                          className="h-4 w-4 rounded border-white/20 bg-transparent"
                        />
                        <span className="text-sm text-slate-300">{formValues[field.name] === "true" ? "Enabled" : "Disabled"}</span>
                      </div>
                    ) : field.type === "categorical" ? (
                      <select
                        aria-label={field.label}
                        value={formValues[field.name] ?? ""}
                        onChange={(event) => handleChange(field.name, event.target.value)}
                        className="w-full rounded-2xl border border-white/10 bg-slate-950/70 px-3 py-2 text-sm text-white outline-none ring-0"
                      >
                        <option value="">Select an option</option>
                        {(field.options ?? []).map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <Input
                        aria-label={field.label}
                        name={field.name}
                        type="number"
                        step="any"
                        inputMode="decimal"
                        value={formValues[field.name] ?? ""}
                        onChange={(event) => handleChange(field.name, event.target.value)}
                        className={formState.errors[field.name] ? "border-red-400/70 bg-red-500/10" : ""}
                      />
                    )}
                    {formState.errors[field.name] ? <p className="text-sm text-red-400">{formState.errors[field.name]}</p> : null}
                  </label>
                ))}
              </div>
            ) : (
              <div className="rounded-2xl border border-dashed border-white/10 bg-white/5 p-4 text-sm text-slate-400">
                {loadingModel ? "Loading feature schema..." : "No feature metadata is available yet for the production model."}
              </div>
            )}

            {error ? (
              <div className="flex items-start gap-2 rounded-2xl border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-300">
                <AlertCircle className="mt-0.5 h-4 w-4 flex-none" />
                <span>{error}</span>
              </div>
            ) : null}

            <div className="flex flex-wrap items-center gap-3 pt-2">
              <Button type="submit" disabled={submitting} className="min-w-[160px] disabled:cursor-not-allowed disabled:opacity-70">
                {submitting ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Predicting...
                  </span>
                ) : (
                  "Predict ETA"
                )}
              </Button>
              <p className="text-sm text-slate-500">Uses POST /api/v1/predict</p>
            </div>
          </form>
        </Card>

        <Card className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="rounded-2xl bg-emerald-500/15 p-2 text-emerald-400">
              <TrendingUp className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Prediction result</h2>
              <p className="text-sm text-slate-400">Live output from the production endpoint.</p>
            </div>
          </div>

          {prediction ? (
            <div className="space-y-4 rounded-2xl border border-emerald-400/20 bg-emerald-500/10 p-4">
              <div>
                <p className="text-sm text-emerald-300">Predicted ETA</p>
                <p className="mt-1 text-3xl font-semibold text-white">{prediction.prediction.toFixed(2)}</p>
              </div>
              <div className="grid gap-3 text-sm text-slate-300 sm:grid-cols-2">
                <div>
                  <p className="text-slate-500">Processing time</p>
                  <p className="font-medium text-white">{prediction.processing_time_ms.toFixed(2)} ms</p>
                </div>
                <div>
                  <p className="text-slate-500">Confidence</p>
                  <p className="font-medium text-white">{explainability?.confidence_score?.toFixed(2) ?? "—"}</p>
                </div>
                <div>
                  <p className="text-slate-500">Model name</p>
                  <p className="font-medium text-white">{prediction.model_name}</p>
                </div>
                <div>
                  <p className="text-slate-500">Version</p>
                  <p className="font-medium text-white">{prediction.model_version}</p>
                </div>
                <div>
                  <p className="text-slate-500">Prediction timestamp</p>
                  <p className="font-medium text-white">{predictionTimestamp ?? "—"}</p>
                </div>
                <div>
                  <p className="text-slate-500">Model metadata</p>
                  <p className="font-medium text-white">{productionModel?.feature_count ?? featureFields.length} features</p>
                </div>
              </div>

              {loadingExplainability ? (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-3 text-sm text-slate-400">Loading explainability insights…</div>
              ) : explainability?.ranked_features?.length ? (
                <div className="space-y-3 rounded-2xl border border-white/10 bg-slate-950/40 p-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-semibold text-white">Top contributing features</p>
                    <Button type="button" className="h-8 px-3 text-xs hover:bg-white/10" onClick={() => router.push("/explainability")}>
                      View Full Explainability
                    </Button>
                  </div>
                  {explainability.ranked_features.slice(0, 5).map((entry, index) => {
                    const contribution = Math.max(0, Math.min(1, entry.contribution ?? 0));
                    return (
                      <div key={`${entry.feature ?? index}`} className="space-y-1">
                        <div className="flex items-center justify-between text-xs text-slate-400">
                          <span>{entry.feature ?? `Feature ${index + 1}`}</span>
                          <span>{contribution.toFixed(2)}</span>
                        </div>
                        <div className="h-2 rounded-full bg-slate-800">
                          <div className="h-2 rounded-full bg-sky-400" style={{ width: `${contribution * 100}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="rounded-2xl border border-dashed border-white/10 bg-white/5 p-4 text-sm text-slate-400">
              Submit a request to view the ETA result here.
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
