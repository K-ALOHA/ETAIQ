"use client";

import { useState } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { Topbar } from "@/components/layout/Topbar";

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <Sidebar
        mobileOpen={mobileMenuOpen}
        onClose={() => setMobileMenuOpen(false)}
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed((value) => !value)}
      />
      <div className={`flex min-h-screen flex-1 flex-col transition-all duration-300 ${collapsed ? "md:pl-20" : "md:pl-72"}`}>
        <Topbar onOpenSidebar={() => setMobileMenuOpen(true)} />
        <main className="flex-1 px-4 pb-10 sm:px-6 lg:px-8">
          {children}
        </main>
      </div>
    </div>
  );
}
