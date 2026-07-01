interface PagePlaceholderProps {
  title: string;
  description: string;
}

export function PagePlaceholder({ title, description }: PagePlaceholderProps) {
  return (
    <div className="flex flex-1 flex-col p-8">
      <header className="mb-8">
        <h1 className="text-3xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          {title}
        </h1>
        <p className="mt-2 max-w-2xl text-zinc-600 dark:text-zinc-400">{description}</p>
      </header>
      <section className="flex flex-1 items-center justify-center rounded-xl border border-dashed border-zinc-300 bg-zinc-50 p-12 dark:border-zinc-700 dark:bg-zinc-900/50">
        <div className="text-center">
          <p className="text-sm font-medium uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Coming Soon
          </p>
          <p className="mt-2 text-zinc-700 dark:text-zinc-300">
            This module will be implemented in a future milestone.
          </p>
        </div>
      </section>
    </div>
  );
}
