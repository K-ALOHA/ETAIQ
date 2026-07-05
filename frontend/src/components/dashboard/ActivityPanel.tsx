import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";

const activities = [
  { title: "Model trained", description: "Production model retrained 3 hours ago." },
  { title: "Production updated", description: "Deployment updated to v2.1.3." },
  { title: "Dataset refreshed", description: "New dataset version imported today." },
  { title: "Prediction service healthy", description: "All endpoints reporting normal status." },
];

export function ActivityPanel() {
  return (
    <Card className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Recent Activity</p>
          <h2 className="mt-2 text-2xl font-semibold text-slate-100">Operational feed</h2>
        </div>
      </div>

      <div className="space-y-3">
        {activities.map((activity) => (
          <motion.div
            key={activity.title}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.25 }}
            className="rounded-3xl border border-white/10 bg-slate-950/90 p-4"
          >
            <p className="text-base font-semibold text-slate-100">{activity.title}</p>
            <p className="mt-1 text-sm text-slate-400">{activity.description}</p>
          </motion.div>
        ))}
      </div>
    </Card>
  );
}
