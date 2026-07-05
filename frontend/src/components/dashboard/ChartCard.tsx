import { ReactNode } from "react";
import { Card } from "@/components/ui/card";

interface ChartCardProps {
  title: string;
  children: ReactNode;
  description: string;
}

export function ChartCard({ title, children, description }: ChartCardProps) {
  return (
    <Card className="space-y-4 p-6">
      <div>
        <p className="text-sm uppercase tracking-[0.24em] text-slate-500">{title}</p>
        <p className="mt-2 text-2xl font-semibold text-white">{description}</p>
      </div>
      <div className="min-h-[320px] rounded-[2rem] bg-slate-950/70 p-4">{children}</div>
    </Card>
  );
}
