import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";

interface DashboardCardProps {
  title: string;
  value: string;
  label: string;
  accent?: string;
}

export function DashboardCard({ title, value, label, accent = "from-sky-500 to-violet-500" }: DashboardCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
    >
      <Card className="group overflow-hidden">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-[0.24em] text-slate-400">{title}</p>
            <p className="mt-4 text-3xl font-semibold text-slate-100">{value}</p>
            <p className="mt-2 text-sm text-slate-400">{label}</p>
          </div>
          <div className={`flex h-12 w-12 items-center justify-center rounded-3xl bg-gradient-to-br ${accent} text-white shadow-lg shadow-slate-950/20`}>✓</div>
        </div>
      </Card>
    </motion.div>
  );
}
