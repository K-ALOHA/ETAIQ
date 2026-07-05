"use client";

import { useState, useEffect, useRef } from "react";
import { ChatInput } from "@/components/assistant/ChatInput";
import { ChatMessage } from "@/components/assistant/ChatMessage";
import { TypingIndicator } from "@/components/assistant/TypingIndicator";
import { WelcomePanel } from "@/components/assistant/WelcomePanel";

interface AssistantMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: string[];
}

interface ChatResponse {
  response: string;
  sources?: string[];
  conversation_id: string;
}

export default function AIAssistantPage() {
  const [messages, setMessages] = useState<AssistantMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const sendMessage = async (message: string) => {
    setError(null);
    const userMessage: AssistantMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: message,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await fetch("/api/v1/assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          conversation_id: conversationId,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to send message");
      }

      const data: ChatResponse = await response.json();

      const assistantMessage: AssistantMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.response,
        timestamp: new Date(),
        sources: data.sources,
      };
      setConversationId(data.conversation_id);
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-[calc(100vh-128px)] flex-col gap-4 px-3 py-4 sm:px-6">
      {messages.length === 0 && !loading ? (
        <WelcomePanel onSelectQuestion={sendMessage} />
      ) : (
        <div ref={scrollContainerRef} className="flex-1 overflow-y-auto rounded-[2rem] border border-white/10 bg-slate-950/50 p-4 sm:p-6">
          <div className="flex flex-col gap-4">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {loading && <TypingIndicator />}
            {error && (
              <div className="mx-auto max-w-md rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                {error}
              </div>
            )}
          </div>
        </div>
      )}
      <ChatInput onSend={sendMessage} disabled={loading} />
    </div>
  );
}
