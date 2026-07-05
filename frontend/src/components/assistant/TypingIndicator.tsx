"use client";

export function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="flex max-w-[80%] items-center gap-3 rounded-3xl border border-white/10 bg-slate-900/85 px-4 py-3 shadow-lg shadow-black/10">
        <div className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 animate-bounce rounded-full bg-slate-400" style={{ animationDelay: "0ms" }} />
          <span className="h-2.5 w-2.5 animate-bounce rounded-full bg-slate-400" style={{ animationDelay: "120ms" }} />
          <span className="h-2.5 w-2.5 animate-bounce rounded-full bg-slate-400" style={{ animationDelay: "240ms" }} />
        </div>
        <span className="text-sm text-slate-400">ETAIQ Assistant is thinking…</span>
      </div>
    </div>
  );
}
