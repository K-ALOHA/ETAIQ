"use client";

import { Sparkles } from "lucide-react";

interface WelcomePanelProps {
  onSelectQuestion: (question: string) => void;
}

const sampleQuestions = [
  "What production model is running?",
  "Show model performance.",
  "Summarize the dataset.",
  "Why is the ETA high?",
  "Predict ETA.",
];

export function WelcomePanel({ onSelectQuestion }: WelcomePanelProps) {
  return (
    <div className="flex h-full items-center justify-center px-3 py-8 sm:px-6">
      <div className="w-full max-w-2xl rounded-[2rem] border border-white/10 bg-slate-950/75 p-8 text-center shadow-2xl shadow-slate-950/30 backdrop-blur-xl">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-3xl bg-gradient-to-br from-sky-500 to-violet-500 text-white shadow-lg shadow-sky-950/30">
          <Sparkles className="h-7 w-7" />
        </div>
        <h2 className="mt-6 text-2xl font-semibold text-white">ETAIQ AI Assistant</h2>
        <p className="mx-auto mt-3 max-w-xl text-sm leading-7 text-slate-400">
          Ask questions about ETA prediction, models, monitoring, training, explainability, datasets, and system health.
        </p>
        <div className="mt-8 grid gap-3 text-left sm:grid-cols-2">
          {sampleQuestions.map((question) => (
            <button
              key={question}
              type="button"
              onClick={() => onSelectQuestion(question)}
              className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-300 transition hover:border-sky-400/30 hover:bg-sky-500/10 hover:text-slate-100"
            >
              {question}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
