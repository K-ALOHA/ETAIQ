import { Card } from "@/components/ui/card";

interface ActivityTimelineProps {
  activities: Array<{ label: string; time: string }>;
}

export function ActivityTimeline({ activities }: ActivityTimelineProps) {
  return (
    <Card className="space-y-6 p-6">
      <div>
        <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Recent Activity</p>
        <h2 className="mt-2 text-2xl font-semibold text-white">Timeline</h2>
      </div>
      <div className="space-y-4">
        {activities.length === 0 ? (
          <p className="text-sm text-slate-400">No recent activity available.</p>
        ) : (
          activities.map((activity) => (
            <div key={`${activity.label}-${activity.time}`} className="rounded-3xl border border-white/10 bg-slate-950/80 p-4">
              <div className="flex items-center justify-between gap-4">
                <p className="text-base font-semibold text-white">{activity.label}</p>
                <span className="text-sm text-slate-500">{activity.time}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </Card>
  );
}
