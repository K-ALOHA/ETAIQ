import { Card } from "@/components/ui/card";

interface StatusItem {
  label: string;
  value: string;
  status: string;
}

interface StatusPanelProps {
  items: StatusItem[];
}

export function StatusPanel({ items }: StatusPanelProps) {
  return (
    <Card className="space-y-4 p-6">
      <div>
        <p className="text-sm uppercase tracking-[0.24em] text-slate-500">System Health</p>
        <h2 className="mt-2 text-2xl font-semibold text-white">Service status</h2>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {items.map((item) => (
          <div key={item.label} className="rounded-3xl border border-white/10 bg-slate-950/80 p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-sm text-slate-400">{item.label}</p>
                <p className="mt-2 text-lg font-semibold text-white">{item.value}</p>
              </div>
              <span className={`rounded-2xl px-3 py-1 text-sm ${item.status === "healthy" ? "bg-emerald-500/15 text-emerald-300" : "bg-rose-500/15 text-rose-300"}`}>
                {item.status}
              </span>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
