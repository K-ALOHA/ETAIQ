"use client";

import { useState } from "react";
import { SendHorizonal } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({ onSend, disabled = false, placeholder = "Ask ETAIQ anything..." }: ChatInputProps) {
  const [value, setValue] = useState("");

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) {
      return;
    }

    onSend(trimmed);
    setValue("");
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="rounded-[1.5rem] border border-white/10 bg-slate-900/70 p-3 shadow-inner shadow-black/20">
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        rows={3}
        disabled={disabled}
        className="min-h-[96px] w-full resize-none border-none bg-transparent px-2 py-2 text-sm leading-7 text-slate-100 outline-none placeholder:text-slate-500 disabled:cursor-not-allowed"
      />
      <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-xs text-slate-500">
          Press Enter to send · Shift + Enter for a new line
        </p>
        <Button
          type="button"
          onClick={submit}
          disabled={disabled || !value.trim()}
          className="h-11 rounded-2xl bg-gradient-to-r from-sky-500 to-violet-500 px-4 text-white shadow-lg shadow-sky-950/30 hover:from-sky-400 hover:to-violet-400 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <SendHorizonal className="mr-2 h-4 w-4" />
          Send
        </Button>
      </div>
    </div>
  );
}
