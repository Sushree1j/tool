import { useEffect, useMemo, useState } from 'react';
import { ArrowUpTrayIcon, CloudArrowDownIcon, ShieldCheckIcon } from '@heroicons/react/24/outline';
import UploadPanel from './components/UploadPanel.jsx';
import FileGallery from './components/FileGallery.jsx';
import StatsStrip from './components/StatsStrip.jsx';
import { formatBytes } from './utils/format.js';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:4000';

export default function App() {
  const [files, setFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const totalUsage = useMemo(
    () => files.reduce((sum, file) => sum + (file.size || 0), 0),
    [files]
  );

  useEffect(() => {
    const controller = new AbortController();

    async function fetchFiles() {
      try {
        setIsLoading(true);
        const res = await fetch(`${API_URL}/api/files`, {
          signal: controller.signal
        });

        if (!res.ok) {
          throw new Error('Unable to load files');
        }

        const data = await res.json();
        setFiles(data.files ?? []);
      } catch (err) {
        if (err.name !== 'AbortError') {
          console.error(err);
          setError('We hit a snag fetching your files. Try again in a moment.');
        }
      } finally {
        setIsLoading(false);
      }
    }

    fetchFiles();

    return () => {
      controller.abort();
    };
  }, []);

  const handleUploaded = (fileMeta) => {
    setFiles((current) => [fileMeta, ...current]);
  };

  return (
    <div className="min-h-screen overflow-x-hidden">
      <header className="relative isolate overflow-hidden">
        <div className="absolute inset-0 -z-10">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.25),_transparent_55%)]" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_bottom,_rgba(236,72,153,0.25),_transparent_60%)]" />
        </div>
        <div className="px-6 pt-12 pb-24 sm:px-12 lg:px-24">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <img src="/logo.svg" alt="NovaShare" className="size-10" />
              <span className="text-xl font-semibold tracking-tight text-white">
                NovaShare
              </span>
            </div>
            <div className="hidden gap-3 sm:flex">
              <button className="rounded-full border border-white/10 bg-white/10 px-4 py-2 text-sm font-medium text-white backdrop-blur transition hover:border-white/30 hover:bg-white/20">
                Sign in
              </button>
              <button className="rounded-full bg-gradient-to-r from-neon via-sapphire to-blush px-4 py-2 text-sm font-semibold text-white shadow-glass transition hover:shadow-[0_22px_60px_rgba(236,72,153,0.35)]">
                Upgrade
              </button>
            </div>
          </div>
          <div className="mx-auto mt-16 max-w-4xl text-center">
            <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-white/70">
              <ShieldCheckIcon className="size-4" /> Enterprise-grade privacy
            </span>
            <h1 className="mt-6 font-display text-4xl font-semibold tracking-tight text-white sm:text-5xl lg:text-6xl">
              Share massive files with breathtaking ease.
            </h1>
            <p className="mt-6 text-base text-white/70 sm:text-lg">
              NovaShare lets you deliver projects up to 10 GB without breaking a sweat. Enjoy glass-smooth uploads, streaming downloads, and links that make clients smile.
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-4 text-sm text-white/80">
              <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2">
                <ArrowUpTrayIcon className="size-4 text-neon" /> 10 GB per transfer
              </div>
              <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2">
                <CloudArrowDownIcon className="size-4 text-blush" /> Fast CDN-backed delivery
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="relative -mt-24 px-6 pb-24 sm:px-12 lg:px-24">
        <div className="mx-auto grid max-w-6xl gap-12 lg:grid-cols-[minmax(0,2fr)_minmax(0,1.1fr)]">
          <UploadPanel apiUrl={API_URL} onUploaded={handleUploaded} />
          <aside className="glass rounded-3xl p-8 shadow-glass">
            <StatsStrip
              totalFiles={files.length}
              totalUsage={formatBytes(totalUsage)}
              isLoading={isLoading}
            />
            <hr className="my-6 border-white/10" />
            <FileGallery
              apiUrl={API_URL}
              files={files}
              isLoading={isLoading}
              onCopyError={setError}
            />
          </aside>
        </div>
      </main>

      <footer className="border-t border-white/10 bg-black/30 py-8 text-center text-xs text-white/40">
        Crafted for teams that deliver heavy ideas. © {new Date().getFullYear()} NovaShare.
      </footer>

      {error && (
        <div className="fixed bottom-6 right-6 w-full max-w-sm rounded-2xl border border-red-500/20 bg-red-500/10 px-5 py-4 text-sm text-red-200 shadow-xl">
          {error}
        </div>
      )}
    </div>
  );
}
