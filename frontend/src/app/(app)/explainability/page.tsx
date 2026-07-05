"use client";

import { useEffect, useMemo, useState } from "react";
import { ArrowUpDown, AlertTriangle, RefreshCw, Sparkles } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card } from "@/components/ui/card";
import { fetchJson } from "@/lib/dashboard";

interface RankedFeatureItem {
  feature_name?: string;
  importance?: number;
  contribution_score?: number;
}

interface LocalExplanationItem {
  feature_name?: string;
  value?: number;
  importance?: number;
  contribution_score?: number;
}

interface ExplainabilityLatestPayload {
  model_name: string;
  version?: string | null;
  latest_prediction_value: number;
  prediction_time: string;
  confidence_score: number;
  explainability_status: string;
  feature_importance: Record<string, number>;
  ranked_features: RankedFeatureItem[];
  local_explanation: LocalExplanationItem[];
  natural_language_explanation: string;
  summary_plot?: string | null;
  waterfall_plot?: string | null;
  metadata_json: string;
  metadata: Record<string, unknown>;
}

type SortKey = "importance" | "contribution" | "feature";

type SortDirection = "asc" | "desc";

function formatMetric(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: value % 1 === 0 ? 0 : 2,
  });
}

function formatTimestamp(value: string) {
  if (!value) {
    return "Pending";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

export default function ExplainabilityPage() {
  const [data, setData] = useState<ExplainabilityLatestPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("importance");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  useEffect(() => {
    let active = true;

    async function loadExplainability() {
      try {
        setLoading(true);
        setError(null);
        const payload = await fetchJson<ExplainabilityLatestPayload>("/api/v1/explainability/latest");
        if (active) {
          setData(payload);
        }
      } catch (err) {
        if (!active) {
          return;
        }

        let message = err instanceof Error ? err.message : "Unable to load explainability data.";
        if (typeof message === "string" && (message.trim().startsWith("<!DOCTYPE") || message.trim().startsWith("<html") || message.includes("<html"))) {
          message = "The explainability service returned an unexpected HTML page.";
        }
        setError(message);
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadExplainability();

    return () => {
      active = false;
    };
  }, []);

  const featureRows = useMemo(() => {
    if (!data) {
      return [];
    }

    const sourceRows = (data.local_explanation?.length ? data.local_explanation : data.ranked_features).map((item, index) => {
      const featureName = item.feature_name || `feature_${index + 1}`;
      const contribution = Number(item.contribution_score ?? item.importance ?? 0);
      const importance = Number(item.importance ?? data.feature_importance?.[featureName] ?? 0);
      return {
        feature: featureName,
        contribution,
        direction: contribution >= 0 ? "+" : "-",
        importance,
      };
    });

    const sortedRows = [...sourceRows].sort((left, right) => {
      const multiplier = sortDirection === "asc" ? 1 : -1;
      if (sortKey === "feature") {
        return left.feature.localeCompare(right.feature) * multiplier;
      }

      if (sortKey === "contribution") {
        return (left.contribution - right.contribution) * multiplier;
      }

      return (left.importance - right.importance) * multiplier;
    });

    const maxContribution = Math.max(...sortedRows.map((row) => Math.abs(row.contribution)), 1);

    return sortedRows.map((row) => ({
      ...row,
      contributionWidth: `${Math.max(8, (Math.abs(row.contribution) / maxContribution) * 100)}%`,
    }));
  }, [data, sortDirection, sortKey]);

  const chartData = useMemo(() => {
    if (!data?.ranked_features?.length) {
      return [];
    }

    return data.ranked_features
      .slice(0, 8)
      .map((item) => ({
        feature: item.feature_name || "feature",
        importance: Number(item.importance ?? 0),
      }));
  }, [data]);

  const topSummary = data
    ? [
        { label: "Latest Prediction", value: formatMetric(data.latest_prediction_value, 2) },
        { label: "Confidence Score", value: `${formatMetric(data.confidence_score, 2)} / 1` },
        { label: "Model", value: data.model_name },
        { label: "Version", value: data.metadata?.version ? String(data.metadata.version) : data.version ?? "—" },
        { label: "Training Date", value: formatTimestamp(String(data.metadata?.trained_at || data.prediction_time)) },
        { label: "Performance Metrics", value: Object.entries(data.metadata?.metrics || {}).map(([name, value]) => `${name}: ${formatMetric(Number(value), 3)}`).join(" • ") || "—" },
      ]
    : [];

  if (loading) {
    return (
      <div className="flex flex-1 flex-col gap-6 p-6 md:p-8">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <div key={index} className="h-28 animate-pulse rounded-3xl border border-white/10 bg-slate-950/70" />
          ))}
        </div>
        <div className="grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
          <div className="h-80 animate-pulse rounded-3xl border border-white/10 bg-slate-950/70" />
          <div className="h-80 animate-pulse rounded-3xl border border-white/10 bg-slate-950/70" />
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex flex-1 items-center justify-center p-6 md:p-8">
        <Card className="max-w-2xl border border-amber-400/20 bg-slate-950/90 p-8 text-center">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-amber-500/10 text-amber-300">
            <AlertTriangle className="h-7 w-7" />
          </div>
          <h1 className="mt-6 text-2xl font-semibold text-white">Explainability workspace temporarily unavailable</h1>
          <p className="mt-3 text-sm leading-6 text-slate-400">{error || "We could not load the latest explainability snapshot."}</p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="mt-6 inline-flex items-center gap-2 rounded-full border border-sky-400/20 bg-sky-500/10 px-4 py-2 text-sm font-medium text-sky-200 transition hover:bg-sky-500/20"
          >
            <RefreshCw className="h-4 w-4" />
            Retry
          </button>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col gap-6 p-6 md:p-8">
      <header className="space-y-2">
        <div className="inline-flex items-center gap-2 rounded-full border border-sky-400/20 bg-sky-500/10 px-3 py-1 text-sm font-medium text-sky-200">
          <Sparkles className="h-4 w-4" />
          Explainability Workspace
        </div>
        <h1 className="text-3xl font-semibold tracking-tight text-white">Real attribution insights from the latest prediction</h1>
        <p className="max-w-2xl text-sm leading-6 text-slate-400">
          Review feature impact, local reasoning, and persisted explainability artifacts without leaving the ETAIQ experience.
        </p>
      </header>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {topSummary.map((item) => (
          <Card key={item.label} className="space-y-2 p-5">
            <p className="text-sm uppercase tracking-[0.24em] text-slate-500">{item.label}</p>
            <p className="text-2xl font-semibold text-white">{item.value}</p>
          </Card>
        ))}
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.25fr_0.75fr]">
        <Card className="space-y-5 p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold text-white">Top Feature Contributions</h2>
              <p className="mt-1 text-sm text-slate-400">Sortable feature attribution impact from the latest prediction.</p>
            </div>
            <button
              type="button"
              onClick={() => {
                setSortDirection((current) => (current === "desc" ? "asc" : "desc"));
              }}
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-300 transition hover:bg-white/10"
            >
              <ArrowUpDown className="h-4 w-4" />
              {sortDirection === "desc" ? "Descending" : "Ascending"}
            </button>
          </div>

          <div className="flex flex-wrap gap-2">
            {(["importance", "contribution", "feature"] as SortKey[]).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => setSortKey(key)}
                className={`rounded-full px-3 py-1.5 text-sm transition ${sortKey === key ? "bg-sky-500/20 text-sky-200" : "bg-white/5 text-slate-400 hover:bg-white/10 hover:text-slate-200"}`}
              >
                {key === "feature" ? "Feature" : key === "contribution" ? "Contribution" : "Importance"}
              </button>
            ))}
          </div>

          <div className="space-y-4">
            {featureRows.map((row) => (
              <div key={row.feature} className="rounded-2xl border border-white/10 bg-slate-900/80 p-4">
                <div className="mb-2 flex items-center justify-between gap-3">
                  <div>
                    <p className="font-medium text-white">{row.feature}</p>
                    <p className="text-sm text-slate-400">{row.direction === "+" ? "Positive" : "Negative"} direction</p>
                  </div>
                  <div className="text-right text-sm text-slate-300">
                    <p>{formatMetric(row.contribution, 3)}</p>
                    <p className="text-slate-500">Impact</p>
                  </div>
                </div>
                <div className="mb-2 h-2 rounded-full bg-slate-800">
                  <div
                    className={`h-2 rounded-full ${row.contribution >= 0 ? "bg-gradient-to-r from-sky-500 to-cyan-400" : "bg-gradient-to-r from-fuchsia-500 to-violet-400"}`}
                    style={{ width: row.contributionWidth }}
                  />
                </div>
                <div className="flex items-center justify-between text-sm text-slate-400">
                  <span>Importance {formatMetric(row.importance, 3)}</span>
                  <span>Contribution {formatMetric(row.contribution, 3)}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card className="space-y-5 p-6">
          <div>
            <h2 className="text-xl font-semibold text-white">Local Explanation</h2>
            <p className="mt-1 text-sm text-slate-400">Model-readable reasoning for the current prediction.</p>
          </div>
          <div className="rounded-2xl border border-sky-400/20 bg-sky-500/10 p-4 text-sm leading-7 text-sky-100">
            {data.natural_language_explanation}
          </div>
          <div className="text-sm text-slate-400">
            <p className="font-medium text-slate-200">Confidence</p>
            <p className="mt-1">{formatMetric(data.confidence_score, 3)} / 1</p>
          </div>
        </Card>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <Card className="space-y-4 p-6">
          <div>
            <h2 className="text-xl font-semibold text-white">Feature Importance</h2>
            <p className="mt-1 text-sm text-slate-400">Global attribution weights from the persisted feature importance artifact.</p>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                <XAxis dataKey="feature" tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={false} tickLine={false} />
                <Tooltip />
                <Bar dataKey="importance" fill="url(#barGradient)" radius={[8, 8, 0, 0]} />
                <defs>
                  <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#38bdf8" />
                    <stop offset="100%" stopColor="#818cf8" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="space-y-4 p-6">
          <div>
            <h2 className="text-xl font-semibold text-white">SHAP Summary</h2>
            <p className="mt-1 text-sm text-slate-400">Visual summary of model-driven attribution.</p>
          </div>
          {data.summary_plot ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={data.summary_plot} alt="SHAP summary plot" className="h-72 w-full rounded-2xl border border-white/10 object-contain" />
          ) : (
            <div className="flex h-72 items-center justify-center rounded-2xl border border-dashed border-white/10 bg-slate-900/70 text-sm text-slate-400">
              SHAP image artifact not available.
            </div>
          )}
        </Card>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <Card className="space-y-4 p-6">
          <div>
            <h2 className="text-xl font-semibold text-white">Waterfall</h2>
            <p className="mt-1 text-sm text-slate-400">Local contribution breakdown for the latest prediction.</p>
          </div>
          {data.waterfall_plot ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={data.waterfall_plot} alt="Waterfall contribution chart" className="h-72 w-full rounded-2xl border border-white/10 object-contain" />
          ) : (
            <div className="flex h-72 items-center justify-center rounded-2xl border border-dashed border-white/10 bg-slate-900/70 text-center text-sm leading-6 text-slate-400">
              SHAP image artifact not available.
            </div>
          )}
        </Card>

        <Card className="space-y-4 p-6">
          <div>
            <h2 className="text-xl font-semibold text-white">Explainability Metadata</h2>
            <p className="mt-1 text-sm text-slate-400">Persisted explainability metadata from the backend artifact bundle.</p>
          </div>
          <details className="rounded-2xl border border-white/10 bg-slate-900/70 p-4">
            <summary className="cursor-pointer text-sm font-medium text-slate-200">Show raw JSON</summary>
            <pre className="mt-4 overflow-x-auto whitespace-pre-wrap text-xs leading-6 text-slate-400">{data.metadata_json}</pre>
          </details>
        </Card>
      </section>
    </div>
  );
}
