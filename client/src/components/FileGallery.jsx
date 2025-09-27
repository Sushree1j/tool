import { useMemo } from 'react';
import { ArrowTopRightOnSquareIcon, ClipboardIcon } from '@heroicons/react/24/outline';
import { formatBytes, formatRelativeDate } from '../utils/format.js';

const skeletons = Array.from({ length: 4 }).map((_, index) => ({ id: `skeleton-${index}` }));

export default function FileGallery({ apiUrl, files, isLoading, onCopyError }) {
  const displayFiles = useMemo(() => {
    if (isLoading && files.length === 0) {
      return skeletons;
    }
    return files;
  }, [files, isLoading]);

  if (files.length === 0 && !isLoading) {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/5 p-6 text-center text-sm text-white/50">
        Your uploads will appear here the moment they finish.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {displayFiles.map((file) => {
        if (!file || !file.id) {
          return (
            <div
              key={file.id}
              className="h-24 animate-pulse rounded-2xl border border-white/5 bg-white/5"
            />
          );
        }

        const shareLink = `${apiUrl}/api/files/${file.code}`;

        return (
          <article
            key={file.id}
            className="group relative overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-6 transition hover:border-white/30 hover:bg-white/10"
          >
            <div className="absolute inset-0 opacity-0 transition group-hover:opacity-100" style={{
              background: 'linear-gradient(135deg, rgba(56,189,248,0.15), rgba(236,72,153,0.2))'
            }} />
            <div className="relative flex flex-col gap-3">
              <div>
                <h3 className="text-sm font-semibold text-white/90">{file.name}</h3>
                <p className="text-xs text-white/50">
                  {formatBytes(file.size)} • uploaded {formatRelativeDate(file.createdAt)}
                </p>
              </div>
              <div className="flex items-center justify-between text-xs text-white/70">
                <span className="truncate">{shareLink}</span>
                <span>{file.downloadCount ?? 0} downloads</span>
              </div>
              <div className="flex gap-2">
                <button
                  className="inline-flex items-center gap-2 rounded-full border border-white/10 px-4 py-2 text-xs font-semibold text-white/80 transition hover:border-white/30 hover:bg-white/10"
                  type="button"
                  onClick={() => {
                    navigator.clipboard.writeText(shareLink).catch(() => onCopyError?.('Unable to copy link. Copy it manually.'));
                  }}
                >
                  <ClipboardIcon className="size-4" /> Copy link
                </button>
                <a
                  className="inline-flex items-center gap-2 rounded-full bg-white/90 px-4 py-2 text-xs font-semibold text-slate-900 transition hover:bg-white"
                  href={`${shareLink}/download`}
                  target="_blank"
                  rel="noreferrer"
                >
                  <ArrowTopRightOnSquareIcon className="size-4" /> Open
                </a>
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );
}
