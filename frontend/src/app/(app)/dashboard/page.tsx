"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Database, Layers, Monitor, Sparkles, TrendingUp } from "lucide-react";
import { ChartCard } from "@/components/dashboard/ChartCard";
import { KpiCard } from "@/components/dashboard/KpiCard";
import { QuickActions } from "@/components/dashboard/QuickActions";
import { ActivityTimeline } from "@/components/dashboard/ActivityTimeline";
import { StatusPanel } from "@/components/dashboard/StatusPanel";
import {
  fetchJson,
  HealthResponse,
  ModelInfoResponse,
  ModelRegistryResponse,
  MonitoringResponse,
} from "@/lib/dashboard";

const topKpiCards = [
  {
    title: "Production Model",
    subtitle: "Active model in production",
    trend: "Status",
    trendLabel: "Production",
    icon: Sparkles,
  },
  {
    title: "MAE",
    subtitle: "Mean absolute error",
    trend: "Metric",
    trendLabel: "Lower is better",
    icon: TrendingUp,
  },
  {
    title: "RMSE",
    subtitle: "Root mean squared error",
    trend: "Metric",
    trendLabel: "Lower is better",
    icon: TrendingUp,
  },
  {
    title: "R²",
    subtitle: "Explained variance",
    trend: "Metric",
    trendLabel: "Higher is better",
    icon: TrendingUp,
  },
  {
    title: "Dataset Size",
    subtitle: "Orders ingested",
    trend: "Data",
    trendLabel: "Available",
    icon: Database,
  },
  {
    title: "Training Samples",
    subtitle: "Records in training set",
    trend: "Data",
    trendLabel: "Available",
    icon: Layers,
  },
  {
    title: "Prediction Latency",
    subtitle: "API response time",
    trend: "Performance",
    trendLabel: "Measured",
    icon: Monitor,
  },
  {
    title: "Model Status",
    subtitle: "Production availability",
    trend: "Status",
    trendLabel: "Real-time",
    icon: Sparkles,
  },
];

export default function DashboardPage() {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [monitoring, setMonitoring] = useState<MonitoringResponse | null>(null);
  const [registryModels, setRegistryModels] = useState<ModelRegistryResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadDashboard() {
      try {
        const [models, monitorData, registryData, serviceHealth] = await Promise.all([
          fetchJson<ModelInfoResponse>("/api/v1/models"),
          fetchJson<MonitoringResponse>("/api/v1/monitoring"),
          fetchJson<ModelRegistryResponse>("/api/v1/models/registry"),
          fetchJson<HealthResponse>("/api/v1/health"),
        ]);

        setModelInfo(models);
        setMonitoring(monitorData);
        setRegistryModels(registryData);
        setHealth(serviceHealth);
      } catch (err) {
        let msg = err instanceof Error ? err.message : "Unable to load dashboard data.";
        if (typeof msg === "string" && (msg.trim().startsWith("<!DOCTYPE") || msg.trim().startsWith("<html") || msg.includes("<html"))) {
          msg = "Server returned an HTML error page. Check backend logs.";
        }
        setError(msg);
      } finally {
        setLoading(false);
      }
    }

    loadDashboard();
  }, []);

  const activityItems = useMemo(() => {
    const registryEntries = (registryModels?.models ?? [])
      .filter((model) => Boolean(model.created_at))
      .slice()
      .sort((left, right) => {
        const leftTime = new Date(left.created_at).getTime();
        const rightTime = new Date(right.created_at).getTime();
        if (leftTime !== rightTime) {
          return rightTime - leftTime;
        }

        const leftIsProduction = (left.status || "").toLowerCase() === "production";
        const rightIsProduction = (right.status || "").toLowerCase() === "production";
        if (leftIsProduction !== rightIsProduction) {
          return leftIsProduction ? -1 : 1;
        }

        return left.model_name.localeCompare(right.model_name);
      });

    const events: Array<{ label: string; time: string; timestamp: number; isProduction: boolean }> = [];
    let pendingArchived: { family: string; count: number; timestamp: number; version: string } | null = null;
    const seenProductionFamilies = new Set<string>();

    const flushPendingArchived = () => {
      if (!pendingArchived) {
        return;
      }

      const label = pendingArchived.count > 1
        ? `${pendingArchived.count} ${pendingArchived.family} models archived`
        : `${pendingArchived.family} v${pendingArchived.version} archived`;

      events.push({
        label,
        time: new Date(pendingArchived.timestamp).toLocaleString(),
        timestamp: pendingArchived.timestamp,
        isProduction: false,
      });
      pendingArchived = null;
    };

    for (const model of registryEntries) {
      const normalizedStatus = (model.status || "").toLowerCase();

      if (normalizedStatus === "production") {
        flushPendingArchived();
        if (!seenProductionFamilies.has(model.model_name)) {
          events.push({
            label: `${model.model_name} v${model.version} promoted to Production`,
            time: new Date(model.created_at).toLocaleString(),
            timestamp: new Date(model.created_at).getTime(),
            isProduction: true,
          });
          seenProductionFamilies.add(model.model_name);
        }
        continue;
      }

      if (normalizedStatus === "archived") {
        const currentPending = pendingArchived;
        if (currentPending && currentPending.family === model.model_name) {
          currentPending.count += 1;
          currentPending.timestamp = Math.min(currentPending.timestamp, new Date(model.created_at).getTime());
          currentPending.version = `${model.version}`;
        } else {
          flushPendingArchived();
          pendingArchived = {
            family: model.model_name,
            count: 1,
            timestamp: new Date(model.created_at).getTime(),
            version: `${model.version}`,
          };
        }
        continue;
      }

      flushPendingArchived();
      events.push({
        label: `${model.model_name} v${model.version} ${model.status.toLowerCase()}`,
        time: new Date(model.created_at).toLocaleString(),
        timestamp: new Date(model.created_at).getTime(),
        isProduction: false,
      });
    }

    flushPendingArchived();

    const sortedEvents = events
      .sort((left, right) => {
        if (left.timestamp !== right.timestamp) {
          return right.timestamp - left.timestamp;
        }

        if (left.isProduction !== right.isProduction) {
          return left.isProduction ? -1 : 1;
        }

        return left.label.localeCompare(right.label);
      })
      .slice(0, 5)
      .map((event) => ({
        label: event.label,
        time: event.time,
      }));

    if (sortedEvents.length) {
      return sortedEvents;
    }

    return [{ label: "No recent activity to display", time: "—" }];
  }, [registryModels]);

  const meaningfulRegistryModels = useMemo(() => {
    if (!registryModels?.models || registryModels.models.length === 0) {
      return [];
    }

    const latestByFamily = new Map<string, (typeof registryModels.models)[number]>();

    for (const model of registryModels.models) {
      const current = latestByFamily.get(model.model_name);
      if (!current || model.version > current.version) {
        latestByFamily.set(model.model_name, model);
      }
    }

    return Array.from(latestByFamily.values())
      .filter((model) => {
        const familyVersions = registryModels.models.filter((candidate) => candidate.model_name === model.model_name);
        return familyVersions.length <= 2;
      })
      .sort((left, right) => left.created_at.localeCompare(right.created_at));
  }, [registryModels]);

  const modelComparisonData = useMemo(() => {
    if (meaningfulRegistryModels.length === 0) {
      return [];
    }

    return meaningfulRegistryModels
      .map((model) => ({
        model: `${model.model_name} v${model.version}`,
        MAE: model.metrics?.mae ?? 0,
        RMSE: model.metrics?.rmse ?? 0,
        R2: model.metrics?.r2 ?? 0,
      }));
  }, [meaningfulRegistryModels]);

  // Extract real metrics for the KPI cards from the backend-selected production model
  const productionModelMetrics = useMemo(() => {
    if (!modelInfo?.models || modelInfo.models.length === 0) {
      return {
        model: "—",
        mae: "—",
        rmse: "—",
        r2: "—",
        mape: "—",
        dataset_size: "—",
        training_samples: "—",
      };
    }

    const currentModelName = modelInfo.current_model || "";
    const currentModelVersion = modelInfo.version;

    const productionModel = modelInfo.models.find((model) => {
      const matchesName = model.model_name && currentModelName.includes(model.model_name);
      const matchesVersion = model.version === currentModelVersion;
      return matchesName && matchesVersion;
    }) || modelInfo.models[0];

    return {
      model: productionModel?.model_name || "—",
      mae: productionModel?.metrics?.mae?.toFixed(4) ?? "—",
      rmse: productionModel?.metrics?.rmse?.toFixed(4) ?? "—",
      r2: productionModel?.metrics?.r2?.toFixed(4) ?? "—",
      mape: productionModel?.metrics?.mape?.toFixed(4) ?? "—",
      dataset_size: typeof productionModel?.dataset_size === "number"
        ? productionModel.dataset_size.toLocaleString()
        : "—",
      training_samples: typeof productionModel?.training_samples === "number"
        ? productionModel.training_samples.toLocaleString()
        : "—",
    };
  }, [modelInfo]);

  const trainingHistoryData = useMemo(() => {
    if (!registryModels?.models || registryModels.models.length === 0) {
      return [];
    }

    return [...registryModels.models]
      .filter((model) => Boolean(model.created_at))
      .sort((left, right) => {
        const leftTime = new Date(left.created_at).getTime();
        const rightTime = new Date(right.created_at).getTime();
        return leftTime - rightTime;
      })
      .map((model) => ({
        created_at: model.created_at,
        timestampLabel: new Date(model.created_at).toLocaleString([], {
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        }),
        model_name: model.model_name,
        version: model.version,
        status: model.status,
        mae: model.metrics?.mae ?? 0,
      }));
  }, [registryModels]);

  const monitoringLatencyValue = useMemo(() => {
    const latencySamples = (monitoring?.records ?? [])
      .map((record) => {
        const candidate = record as typeof record & Record<string, unknown>;
        const value = candidate.latency_ms ?? candidate.prediction_latency_ms ?? candidate.latency;
        return typeof value === "number" ? value : null;
      })
      .filter((value): value is number => typeof value === "number");

    if (latencySamples.length === 0) {
      return "—";
    }

    const averageLatency = latencySamples.reduce((sum, value) => sum + value, 0) / latencySamples.length;
    return `${averageLatency.toFixed(2)} ms`;
  }, [monitoring]);

  const values = useMemo(() => [
    modelInfo?.current_model || "—",
    productionModelMetrics.mae,
    productionModelMetrics.rmse,
    productionModelMetrics.r2,
    productionModelMetrics.dataset_size,
    productionModelMetrics.training_samples,
    monitoringLatencyValue,
    health?.model_loaded ? "Healthy" : "—",
  ], [modelInfo, productionModelMetrics, monitoringLatencyValue, health]);

  const healthItems = useMemo(
    () => [
      { label: "Backend", value: health?.status ?? (loading ? "Loading" : "Offline"), status: health?.status === "healthy" ? "healthy" : "offline" },
      { label: "Monitoring", value: monitoring ? "Active" : loading ? "Loading" : "Unavailable", status: monitoring ? "healthy" : "offline" },
      { label: "Registry", value: modelInfo ? "Healthy" : loading ? "Loading" : "Degraded", status: modelInfo ? "healthy" : "offline" },
      { label: "Dataset", value: monitoring ? "Loaded" : loading ? "Loading" : "Unavailable", status: monitoring ? "healthy" : "offline" },
      { label: "Model", value: modelInfo ? "Production" : loading ? "Loading" : "Unavailable", status: modelInfo ? "healthy" : "offline" },
    ],
    [health, modelInfo, monitoring, loading]
  );

  return (
    <div className="mx-auto flex max-w-[1600px] flex-col gap-8 pb-8">
      <section className="rounded-[2rem] border border-white/10 bg-slate-950/90 p-6 shadow-2xl shadow-black/20 backdrop-blur-xl">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.32em] text-sky-300/70">AI Operations</p>
            <h1 className="mt-3 text-4xl font-semibold text-white">Production dashboard</h1>
            <p className="mt-2 max-w-2xl text-slate-400">Real-time model performance, activity streams, and system health for ETAIQ.</p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded-3xl bg-white/5 px-4 py-3 text-sm font-medium text-slate-100">Production model active</span>
            <span className="rounded-3xl bg-emerald-500/10 px-4 py-3 text-sm font-medium text-emerald-300">Stable release</span>
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-4">
        {topKpiCards.map((card, index) => (
          <KpiCard
            key={card.title}
            icon={card.icon}
            title={card.title}
            value={values[index]}
            subtitle={card.subtitle}
            trend={card.trend}
            trendLabel={card.trendLabel}
            loading={loading}
            error={error || undefined}
          />
        ))}
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.6fr_1fr]">
        <div className="space-y-6">
          <ChartCard title="Model Comparison" description="Performance across candidate models">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={modelComparisonData} margin={{ top: 24, right: 32, bottom: 12, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                <XAxis dataKey="model" tick={{ fill: "#94a3b8" }} />
                <YAxis tick={{ fill: "#94a3b8" }} />
                <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid rgba(148,163,184,0.15)", borderRadius: 20 }} />
                <Bar dataKey="MAE" fill="#38bdf8" radius={[10, 10, 0, 0]} />
                <Bar dataKey="RMSE" fill="#8b5cf6" radius={[10, 10, 0, 0]} />
                <Bar dataKey="R2" fill="#34d399" radius={[10, 10, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Training History" description="Model evolution over time">
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={trainingHistoryData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                <XAxis
                  dataKey="created_at"
                  tick={{ fill: "#94a3b8" }}
                  tickFormatter={(value: string) => new Date(value).toLocaleString([], {
                    month: "short",
                    day: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                />
                <YAxis tick={{ fill: "#94a3b8" }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", border: "1px solid rgba(148,163,184,0.15)", borderRadius: 20 }}
                  formatter={(value) => [typeof value === "number" ? value : Number(value ?? 0), "MAE"]}
                  labelFormatter={(label) => {
                    const point = trainingHistoryData.find((entry) => entry.created_at === label);
                    return point ? `${point.model_name} v${point.version}` : label;
                  }}
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) {
                      return null;
                    }

                    const point = trainingHistoryData.find((entry) => entry.created_at === label);
                    if (!point) {
                      return null;
                    }

                    return (
                      <div className="rounded-2xl border border-white/10 bg-slate-950/95 p-3 text-sm text-slate-100 shadow-2xl shadow-black/20">
                        <p className="font-semibold">{`${point.model_name} v${point.version}`}</p>
                        <p className="mt-1 text-slate-400">Status: {point.status}</p>
                        <p className="mt-1 text-slate-400">Created: {point.timestampLabel}</p>
                        <p className="mt-1 text-slate-400">MAE: {point.mae}</p>
                      </div>
                    );
                  }}
                />
                <Line type="monotone" dataKey="mae" stroke="#38bdf8" strokeWidth={3} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        <div className="space-y-6">
          <QuickActions />
          <ActivityTimeline activities={activityItems} />
          <StatusPanel items={healthItems} />
        </div>
      </section>
    </div>
  );
}
