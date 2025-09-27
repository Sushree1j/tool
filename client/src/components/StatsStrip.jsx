import { ServerStackIcon, SparklesIcon } from '@heroicons/react/24/outline';

export default function StatsStrip({ totalFiles, totalUsage, isLoading }) {
  return (
    <div className="flex flex-col gap-6 rounded-2xl border border-white/10 bg-white/5 p-6">
      <div className="flex items-center gap-4 text-white">
        <div className="rounded-2xl bg-gradient-to-br from-neon/30 via-sapphire/30 to-blush/30 p-3">
          <ServerStackIcon className="size-8 text-neon" />
        </div>
        <div>
          <p className="text-xs uppercase tracking-[0.4em] text-white/50">Storage used</p>
          <p className="mt-1 text-2xl font-semibold text-white/90">{isLoading ? '—' : totalUsage}</p>
        </div>
      </div>
      <div className="flex items-center gap-4 rounded-2xl border border-white/5 bg-black/30 p-4">
        <div className="rounded-full bg-white/10 p-2">
          <SparklesIcon className="size-6 text-blush" />
        </div>
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-white/50">Files shared</p>
          <p className="mt-1 text-xl font-semibold text-white/80">{isLoading ? '—' : totalFiles}</p>
        </div>
      </div>
      <p className="text-xs text-white/50">
        NovaShare stores your transfers with AES-256 encryption in an isolated vault. Files auto-expire after 30 days of inactivity to keep things lean.
      </p>
    </div>
  );
}
