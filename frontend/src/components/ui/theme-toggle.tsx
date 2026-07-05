"use client";

import { useState } from "react";
import { useTheme } from "next-themes";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";

export function ThemeToggle() {
  const [mounted, setMounted] = useState(false);
  const { theme, setTheme } = useTheme();

  if (!mounted) {
    return (
      <Button
        type="button"
        aria-label="Toggle theme"
        className="h-11 w-11 rounded-2xl"
        disabled
      />
    );
  }

  return (
    <Button
      type="button"
      aria-label="Toggle theme"
      onClick={() => {
        setMounted(true);
        setTheme(theme === "dark" ? "light" : "dark");
      }}
      className="h-11 w-11 rounded-2xl bg-slate-900/85 p-0 text-slate-100"
    >
      {theme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
    </Button>
  );
}
