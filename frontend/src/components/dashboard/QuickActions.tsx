"use client";

import { motion } from "framer-motion";
import { ArrowRight, Cpu, Hammer, Layers, ShieldCheck, Sparkles } from "lucide-react";

const actions = [
  { label: "Predict ETA", icon: Sparkles, variant: "from-sky-500 to-cyan-500" },
  { label: "Train Model", icon: Cpu, variant: "from-violet-500 to-pink-500" },
  { label: "Model Registry", icon: Layers, variant: "from-emerald-500 to-teal-500" },
  { label: "Monitoring", icon: ShieldCheck, variant: "from-amber-500 to-orange-500" },
  { label: "Explainability", icon: Hammer, variant: "from-fuchsia-500 to-violet-500" },
];

export function QuickActions() {
  return (
    <div className="rounded-3xl border border-white/10 bg-slate-950/90 p-6 shadow-2xl shadow-black/20">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Quick Actions</p>
          <h2 className="mt-2 text-2xl font-semibold text-white">Take action</h2>
        </div>
      </div>
      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        {actions.map((action) => (
          <motion.button
            key={action.label}
            whileHover={{ y: -2 }}
            className="flex items-center justify-between rounded-3xl border border-white/10 bg-slate-950/80 px-4 py-4 text-left transition hover:bg-slate-900/90"
          >
            <div className="flex items-center gap-3">
              <span className={`flex h-12 w-12 items-center justify-center rounded-3xl bg-gradient-to-br ${action.variant} text-white`}>
                <action.icon className="h-5 w-5" />
              </span>
              <div>
                <p className="text-base font-semibold text-white">{action.label}</p>
                <p className="text-sm text-slate-400">Open the relevant workspace</p>
              </div>
            </div>
            <ArrowRight className="h-5 w-5 text-slate-400" />
          </motion.button>
        ))}
      </div>
    </div>
  );
}
