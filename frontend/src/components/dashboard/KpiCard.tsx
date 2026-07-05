"use client";

import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";
import { Card } from "@/components/ui/card";
import { AnimatedNumber } from "@/components/dashboard/AnimatedNumber";

interface KpiCardProps {
  icon: LucideIcon;
  title: string;
  value: string | number;
  subtitle: string;
  trend: string;
  trendLabel: string;
  loading?: boolean;
  error?: string;
}

export function KpiCard({ icon: Icon, title, value, subtitle, trend, trendLabel, loading, error }: KpiCardProps) {
  return (
    <motion.div whileHover={{ y: -4 }} className="rounded-3xl transition-shadow duration-300 hover:shadow-[0_20px_80px_rgba(15,23,42,0.45)]">
      <Card className="flex h-full flex-col justify-between gap-4 p-6">
        <div className="flex items-center justify-between gap-4">
          <div className="rounded-3xl bg-slate-900/80 p-3 text-sky-300 shadow-inner shadow-black/10">
            <Icon className="h-6 w-6" />
          </div>
          <p className="text-xs uppercase tracking-[0.24em] text-slate-500">{title}</p>
        </div>

        <div className="space-y-2">
          {loading ? (
            <div className="h-12 w-3/4 animate-pulse rounded-2xl bg-slate-800/80" />
          ) : error ? (
            <p className="text-sm text-rose-400">{error}</p>
          ) : (
            <p className="text-3xl font-semibold text-white">
              {typeof value === "number" ? <AnimatedNumber value={value} /> : value}
            </p>
          )}
          <p className="text-sm text-slate-400">{subtitle}</p>
        </div>

        <div className="rounded-3xl bg-slate-900/80 px-4 py-3 text-sm text-slate-300">
          <div className="flex items-center justify-between gap-2">
            <span>{trend}</span>
            <span className="font-semibold text-slate-100">{trendLabel}</span>
          </div>
        </div>
      </Card>
    </motion.div>
  );
}
