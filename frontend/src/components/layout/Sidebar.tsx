"use client";

import Link from "next/link";
import { Menu, Home, Truck, Activity, Cpu, Layers, Database, Monitor, Search, Sparkles, Settings } from "lucide-react";
import { SidebarNavItem } from "@/components/layout/SidebarNavItem";
import { Button } from "@/components/ui/button";

interface SidebarProps {
  mobileOpen: boolean;
  onClose: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

const navigation = [
  { label: "Dashboard", href: "/dashboard", icon: Home },
  { label: "Predict ETA", href: "/prediction", icon: Truck },
  { label: "Model Performance", href: "/model-performance", icon: Activity },
  { label: "Training", href: "/training", icon: Cpu },
  { label: "Model Registry", href: "/model-registry", icon: Layers },
  { label: "Monitoring", href: "/monitoring", icon: Monitor },
  { label: "Explainability", href: "/explainability", icon: Search },
  { label: "Dataset Explorer", href: "/dataset-explorer", icon: Database },
  { label: "AI Assistant", href: "/ai-assistant", icon: Sparkles },
  { label: "Settings", href: "/settings", icon: Settings },
] as const;

export function Sidebar({ mobileOpen, onClose, collapsed, onToggleCollapse }: SidebarProps) {
  return (
    <>
      <div
        className={`fixed inset-y-0 left-0 z-50 w-full bg-slate-950/95 px-3 py-4 shadow-2xl shadow-black/40 backdrop-blur-xl transition-transform duration-300 md:hidden ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between px-2">
          <Link href="/dashboard" className="text-lg font-semibold tracking-tight text-white">
            ETAIQ
          </Link>
          <Button type="button" className="h-11 w-11 p-0" onClick={onClose}>
            <Menu className="h-5 w-5" />
          </Button>
        </div>
        <nav className="mt-6 flex flex-col gap-2 px-1">
          {navigation.map((item) => (
            <SidebarNavItem key={item.href} href={item.href} label={item.label} icon={item.icon} />
          ))}
        </nav>
      </div>

      <aside className={`hidden h-full min-h-screen flex-col gap-6 border-r border-white/10 bg-slate-950/95 px-4 py-6 shadow-2xl shadow-black/20 backdrop-blur-xl md:flex ${collapsed ? "w-20" : "w-72"}`}>
        <div className="flex items-center justify-between gap-3 px-2">
          <Link href="/dashboard" className="flex items-center gap-2 text-white">
            <div className="flex h-11 w-11 items-center justify-center rounded-3xl bg-gradient-to-br from-sky-500 to-violet-500 text-sm font-semibold shadow-lg shadow-slate-950/30">
              AI
            </div>
            {!collapsed && (
              <div>
                <p className="text-lg font-semibold tracking-tight">ETAIQ</p>
                <p className="text-xs text-slate-400">Deliver smarter ETA</p>
              </div>
            )}
          </Link>
          <Button type="button" className="h-11 w-11 p-0" onClick={onToggleCollapse}>
            <Menu className="h-5 w-5" />
          </Button>
        </div>

        <nav className="flex flex-1 flex-col gap-2 px-1">
          {navigation.map((item) => (
            <SidebarNavItem key={item.href} href={item.href} label={item.label} icon={item.icon} collapsed={collapsed} />
          ))}
        </nav>

        {!collapsed && (
          <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-xs text-slate-300">
            <p className="font-semibold text-slate-100">Premium AI Insights</p>
            <p className="mt-2 text-slate-500">Manage models, datasets, monitoring, and governance in one unified workspace.</p>
          </div>
        )}
      </aside>
    </>
  );
}
