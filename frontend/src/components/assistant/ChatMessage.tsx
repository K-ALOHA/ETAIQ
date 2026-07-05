"use client";

import { Bot, User } from "lucide-react";

interface AssistantMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string | Date;
  sources?: string[];
}

interface ChatMessageProps {
  message: AssistantMessage;
}

const formatTimestamp = (value: string | Date) => {
  const date = value instanceof Date ? value : new Date(value);
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
};

const formatInlineMarkdown = (value: string) => {
  const parts = value.split(/(\*\*[^*]+\*\*|`[^`]+`|_[^_]+_)/g).filter(Boolean);

  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={`${part}-${index}`} className="rounded bg-slate-800/80 px-1.5 py-0.5 text-sm text-sky-200">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith("_") && part.endsWith("_")) {
      return <em key={`${part}-${index}`}>{part.slice(1, -1)}</em>;
    }
    return <span key={`${part}-${index}`}>{part}</span>;
  });
};

const renderMarkdown = (value: string) => {
  const blocks = value
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);

  return blocks.map((block, blockIndex) => {
    const lines = block.split("\n");
    const isList = lines.every((line) => /^[-*]\s+/.test(line.trim()));

    if (isList) {
      return (
        <ul key={`${blockIndex}`} className="ml-4 list-disc space-y-1">
          {lines.map((line, lineIndex) => (
            <li key={`${blockIndex}-${lineIndex}`}>{formatInlineMarkdown(line.replace(/^[-*]\s+/, ""))}</li>
          ))}
        </ul>
      );
    }

    if (lines[0]?.startsWith("# ")) {
      return (
        <h3 key={`${blockIndex}`} className="text-sm font-semibold text-slate-100">
          {formatInlineMarkdown(lines[0].slice(2))}
        </h3>
      );
    }

    return (
      <div key={`${blockIndex}`} className="space-y-2">
        {lines.map((line, lineIndex) => (
          <p key={`${blockIndex}-${lineIndex}`} className="leading-7 text-slate-200">
            {line ? formatInlineMarkdown(line) : <span className="block h-2" />}
          </p>
        ))}
      </div>
    );
  });
};

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`flex max-w-[92%] gap-3 sm:max-w-[80%] ${isUser ? "flex-row-reverse" : "flex-row"}`}>
        <div
          className={`mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl border ${
            isUser
              ? "border-sky-400/30 bg-sky-500/15 text-sky-200"
              : "border-white/10 bg-slate-900/90 text-slate-200"
          }`}
        >
          {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
        </div>
        <div className={`flex min-w-0 flex-col ${isUser ? "items-end" : "items-start"}`}>
          <div className={`rounded-3xl px-4 py-3 shadow-lg shadow-black/10 ${isUser ? "bg-sky-500/15 text-slate-100" : "bg-slate-900/85 text-slate-100"}`}>
            <div className="prose prose-invert max-w-none prose-p:my-1 prose-ul:my-2 prose-li:my-0.5 prose-headings:text-slate-50 prose-code:text-sky-200">
              {renderMarkdown(message.content)}
            </div>
          </div>
          <div className={`mt-2 flex items-center gap-2 text-[11px] uppercase tracking-[0.24em] text-slate-500 ${isUser ? "justify-end" : "justify-start"}`}>
            <span>{formatTimestamp(message.timestamp)}</span>
            {!isUser && message.sources && message.sources.length > 0 && (
              <span className="rounded-full border border-white/10 bg-white/5 px-2 py-1 text-[10px] tracking-[0.2em] text-slate-400">
                {message.sources.length} source{message.sources.length === 1 ? "" : "s"}
              </span>
            )}
          </div>
          {!isUser && message.sources && message.sources.length > 0 ? (
            <div className="mt-2 flex flex-wrap gap-2">
              {message.sources.map((source) => (
                <span key={source} className="rounded-full border border-sky-400/20 bg-sky-500/10 px-2.5 py-1 text-[11px] font-medium text-sky-200">
                  {source}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
