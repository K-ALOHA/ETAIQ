import { forwardRef, type InputHTMLAttributes } from "react";

type InputProps = InputHTMLAttributes<HTMLInputElement>;

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className = "", ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={`min-h-[44px] w-full rounded-2xl border border-transparent bg-slate-950/10 px-3 text-sm text-slate-100 outline-none placeholder:text-slate-500 transition focus:border-sky-400 focus:bg-slate-950/20 focus:ring-2 focus:ring-sky-400/20 ${className}`}
        {...props}
      />
    );
  }
);

Input.displayName = "Input";
