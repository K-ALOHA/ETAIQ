"use client";

import { useMemo } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Bell, Menu, Search, UserCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ThemeToggle } from "@/components/ui/theme-toggle";

const pageTitles: Record<string, string> = {
  dashboard: "Dashboard",
  "predict-eta": "Predict ETA",
  "model-performance": "Model Performance",
  training: "Training",
  "model-registry": "Model Registry",
  monitoring: "Monitoring",
  explainability: "Explainability",
  "dataset-explorer": "Dataset Explorer",
  "ai-assistant": "AI Assistant",
  settings: "Settings",
};

interface TopbarProps {
  onOpenSidebar: () => void;
}

export function Topbar({ onOpenSidebar }: TopbarProps) {
  const pathname = usePathname();

  const breadcrumbs = useMemo(() => {
    const segments = pathname.split("/").filter(Boolean);
    if (segments.length === 0) {
      return ["Dashboard"];
    }

    return segments.map((segment) => pageTitles[segment] ?? segment.replace(/-/g, " ")); 
  }, [pathname]);

  return (
    <div className="sticky top-0 z-30 border-b border-white/10 bg-slate-950/85 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1600px] flex-col gap-4 px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Button
              type="button"
              className="h-11 w-11 rounded-2xl bg-slate-900/85 p-0 text-slate-100 shadow-xl shadow-black/20 md:hidden"
              onClick={onOpenSidebar}
            >
              <Menu className="h-5 w-5" />
            </Button>

            <div className="hidden items-center gap-3 rounded-3xl border border-white/10 bg-slate-950/90 px-4 py-3 shadow-2xl shadow-black/20 md:flex">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-500 to-violet-500 text-white shadow-lg shadow-slate-950/20">
                AI
              </div>
              <div>
                <p className="text-sm font-medium text-slate-300">ETAIQ</p>
                <p className="text-xs text-slate-500">AI Intelligence Dashboard</p>
              </div>
            </div>
          </div>

          <div className="flex flex-1 items-center justify-end gap-3 sm:justify-between">
            <div className="hidden flex-1 md:flex md:max-w-xl md:items-center md:gap-3">
              <div className="relative flex w-full items-center rounded-2xl border border-white/10 bg-slate-950/80 px-3 py-2 shadow-sm shadow-black/10">
                <Search className="h-5 w-5 text-slate-400" />
                <Input
                  className="border-none bg-transparent px-3 text-sm text-slate-100 placeholder:text-slate-500 focus:ring-0"
                  placeholder="Search ETAIQ"
                />
              </div>
            </div>

            <div className="flex items-center gap-2">
              <ThemeToggle />
              <Button type="button" className="h-11 w-11 rounded-2xl bg-slate-900/85 p-0 text-slate-200 shadow-xl shadow-black/20">
                <Bell className="h-5 w-5" />
              </Button>
              <Link href="/settings" className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/5 text-slate-100 shadow-xl shadow-black/20 ring-1 ring-white/10 transition hover:bg-white/10">
                <UserCircle className="h-6 w-6" />
              </Link>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.24em] text-slate-500 sm:gap-3">
          {breadcrumbs.map((crumb, index) => (
            <span key={`${crumb}-${index}`} className="inline-flex items-center gap-2">
              {index > 0 && <span className="text-slate-600">/</span>}
              <span>{crumb}</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
