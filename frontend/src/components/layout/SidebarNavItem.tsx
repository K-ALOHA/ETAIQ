import Link from "next/link";
import { usePathname } from "next/navigation";
import { LucideIcon } from "lucide-react";

interface SidebarNavItemProps {
  href: string;
  label: string;
  icon: LucideIcon;
  collapsed?: boolean;
}

export function SidebarNavItem({ href, label, icon: Icon, collapsed = false }: SidebarNavItemProps) {
  const pathname = usePathname();
  const active = pathname === href || pathname.startsWith(`${href}/`);

  return (
    <Link
      href={href}
      className={`group flex items-center gap-3 rounded-3xl px-4 py-3 text-sm font-medium transition-all duration-200 ${
        active
          ? "bg-sky-500/15 text-sky-100 shadow-lg shadow-sky-500/10"
          : "text-slate-300 hover:bg-white/5 hover:text-white"
      }`}
    >
      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/5 text-slate-300 transition group-hover:bg-sky-500/10 group-hover:text-sky-200">
        <Icon className="h-5 w-5" />
      </div>
      <span className={collapsed ? "sr-only" : ""}>{label}</span>
    </Link>
  );
}
