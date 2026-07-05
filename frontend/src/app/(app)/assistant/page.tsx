"use client";

import { useEffect, useRef, useState } from "react";
import { Bot } from "lucide-react";
import { ChatInput } from "@/components/assistant/ChatInput";
import { ChatMessage } from "@/components/assistant/ChatMessage";
import { TypingIndicator } from "@/components/assistant/TypingIndicator";
import { WelcomePanel } from "@/components/assistant/WelcomePanel";

interface AssistantMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  sources?: string[];
}

export default function AssistantPage() {
  const [messages, setMessages] = useState<AssistantMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [retryMessage, setRetryMessage] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading, autoScroll]);

  const handleSend = async (content: string) => {
    const trimmed = content.trim();
    if (!trimmed || loading) {
      return;
    }

    const userMessage: AssistantMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
      timestamp: new Date().toISOString(),
    };

    setMessages((current) => [...current, userMessage]);
    setLoading(true);
    setError(null);
    setRetryMessage(trimmed);

    try {
      const response = await fetch("/api/v1/assistant/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          conversation_id: conversationId ?? null,
          message: trimmed,
        }),
      });

      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(payload?.detail || "request failed");
      }

      const assistantMessage: AssistantMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: payload?.response || "I’m here to help with ETAIQ.",
        timestamp: new Date().toISOString(),
        sources: Array.isArray(payload?.sources) ? payload.sources : [],
      };

      setConversationId(payload?.conversation_id || null);
      setMessages((current) => [...current, assistantMessage]);
    } catch (err) {
      const backendUnavailable = err instanceof Error && err.message === "gemini unavailable";
      setError(
        backendUnavailable ? "Unable to contact ETAIQ Assistant." : "We couldn’t complete that request. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleRetry = () => {
    if (retryMessage) {
      void handleSend(retryMessage);
    }
  };

  return (
    <div className="flex min-h-[70vh] flex-col rounded-[2rem] border border-white/10 bg-slate-950/70 p-4 shadow-2xl shadow-black/20 backdrop-blur-xl sm:p-6 lg:p-8">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-500 to-violet-500 text-white shadow-lg shadow-slate-950/30">
              <Bot className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-white">AI Assistant</h1>
              <p className="text-sm text-slate-400">
                Ask questions about ETA prediction, models, monitoring, training, explainability, datasets, and system health.
              </p>
            </div>
          </div>
        </div>
        <div className="rounded-full border border-emerald-400/20 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-300">
          Online
        </div>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto px-1 py-2"
        onScroll={(event) => {
          const target = event.currentTarget;
          const isNearBottom = target.scrollHeight - target.scrollTop - target.clientHeight < 40;
          setAutoScroll(isNearBottom);
        }}
      >
        {messages.length === 0 && !loading ? (
          <WelcomePanel onSelectQuestion={handleSend} />
        ) : (
          <div className="mx-auto flex w-full max-w-4xl flex-col gap-4">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {loading ? <TypingIndicator /> : null}
          </div>
        )}
      </div>

      {error ? (
        <div className="mt-4 rounded-2xl border border-rose-400/20 bg-rose-500/10 p-4 text-sm text-rose-200">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <span>{error}</span>
            {retryMessage ? (
              <button
                type="button"
                onClick={handleRetry}
                className="rounded-full border border-rose-400/20 bg-rose-500/15 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.2em] text-rose-100 transition hover:bg-rose-500/20"
              >
                Retry
              </button>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="mt-4">
        <ChatInput onSend={handleSend} disabled={loading} />
      </div>
    </div>
  );
}
