import { useCallback, useRef, useState } from 'react';
import { CheckCircleIcon, DocumentArrowUpIcon, LockClosedIcon } from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { formatBytes } from '../utils/format.js';

const TEN_GB = 10 * 1024 * 1024 * 1024;

const benefits = [
  { icon: LockClosedIcon, label: 'Encrypted at rest' },
  { icon: CheckCircleIcon, label: 'Zero tracking' },
  { icon: DocumentArrowUpIcon, label: 'Resume-friendly uploads' }
];

export default function UploadPanel({ apiUrl, onUploaded }) {
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Drop a file to begin.');
  const [isUploading, setIsUploading] = useState(false);
  const [link, setLink] = useState('');
  const [error, setError] = useState('');
  const inputRef = useRef(null);

  const reset = useCallback(() => {
    setProgress(0);
    setStatus('Drop a file to begin.');
    setIsUploading(false);
    setLink('');
    setError('');
  }, []);

  const handleFile = useCallback(
    (selected) => {
      const item = selected?.[0];
      if (!item) return;

      if (item.size > TEN_GB) {
        setError('That file is larger than 10 GB. Try something smaller.');
        reset();
        return;
      }

      setFile(item);
      setStatus(`${item.name} • ${formatBytes(item.size)}`);
      setProgress(0);
      setError('');
      setLink('');
    },
    [reset]
  );

  const handleSelect = useCallback((event) => {
    handleFile(event.target.files);
  }, [handleFile]);

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      handleFile(event.dataTransfer?.files);
    },
    [handleFile]
  );

  const handleDragOver = useCallback((event) => {
    event.preventDefault();
  }, []);

  const upload = useCallback(() => {
    if (!file) return;

    setIsUploading(true);
    setStatus('Uploading…');
    setError('');
    setLink('');

    const formData = new FormData();
    formData.append('file', file);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${apiUrl}/api/files`);

    xhr.upload.addEventListener('progress', (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.round((event.loaded / event.total) * 100);
      setProgress(percent);
      setStatus(`Uploading… ${percent}%`);
    });

    xhr.addEventListener('load', () => {
      setIsUploading(false);
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const payload = JSON.parse(xhr.responseText);
          setStatus('Upload complete');
          setProgress(100);
          setLink(payload.link);
          onUploaded?.(payload.file);
        } catch (parseError) {
          console.error(parseError);
          setError('Upload succeeded but the server sent an unexpected response.');
        }
      } else {
        console.error(xhr.responseText);
        const message = xhr.status === 413
          ? 'This file exceeds the 10 GB limit. Please compress or split it.'
          : 'Something went wrong while uploading. Try again in a moment.';
        setError(message);
        setStatus('Upload failed');
      }
    });

    xhr.addEventListener('error', () => {
      setIsUploading(false);
      setError('Network error interrupted the upload.');
      setStatus('Upload failed');
    });

    xhr.send(formData);
  }, [apiUrl, file, onUploaded]);

  const clearSelection = useCallback(() => {
    setFile(null);
    reset();
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  }, [reset]);

  return (
    <section className="glass grid h-max gap-8 rounded-3xl p-10 shadow-glass">
      <div
        className={clsx(
          'flex flex-col items-center justify-center rounded-3xl border border-dashed border-white/20 px-6 py-16 text-center transition',
          file ? 'border-neon/70 bg-neon/5' : 'hover:border-white/40 hover:bg-white/5'
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        role="button"
        tabIndex={0}
      >
        <div className="mx-auto flex size-16 items-center justify-center rounded-full bg-gradient-to-br from-neon/30 via-sapphire/30 to-blush/30">
          <DocumentArrowUpIcon className="size-8 text-neon" />
        </div>
        <h2 className="mt-6 text-2xl font-semibold text-white">
          Drop your file here
        </h2>
        <p className="mt-2 max-w-md text-sm text-white/60">
          Supports single transfers up to 10 GB. Drag and drop or use the button below to pick a file securely.
        </p>
        <button
          className="mt-6 inline-flex items-center gap-2 rounded-full bg-white/90 px-6 py-2 text-sm font-semibold text-slate-900 transition hover:bg-white"
          onClick={() => inputRef.current?.click()}
          type="button"
          disabled={isUploading}
        >
          Choose file
        </button>
        <input
          ref={inputRef}
          type="file"
          className="sr-only"
          onChange={handleSelect}
        />
      </div>

      <div className="rounded-2xl border border-white/10 bg-black/30 px-6 py-5">
        <div className="flex flex-col gap-2 text-left">
          <span className="text-xs uppercase tracking-[0.3em] text-white/50">Status</span>
          <p className="text-sm font-medium text-white/90">{status}</p>
          {file && (
            <p className="text-xs text-white/50">{formatBytes(file.size)}</p>
          )}
          {Boolean(progress) && (
            <div className="mt-3 h-1.5 w-full rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-gradient-to-r from-neon via-sapphire to-blush transition-all"
                style={{ width: `${Math.max(progress, 4)}%` }}
              />
            </div>
          )}
        </div>
      </div>

      {link && (
        <div className="rounded-2xl border border-neon/30 bg-neon/10 px-6 py-5 text-left text-sm text-white/90">
          <p className="font-semibold">Shareable link</p>
          <p className="mt-1 break-all text-xs text-white/70">{link}</p>
          <button
            className="mt-3 rounded-full border border-white/10 px-4 py-2 text-xs font-semibold text-white/80 transition hover:border-white/30 hover:bg-white/10"
            onClick={() => {
              navigator.clipboard.writeText(link).catch(() => setError('Unable to copy link. Copy it manually.'));
            }}
            type="button"
          >
            Copy link
          </button>
        </div>
      )}

      {error && (
        <div className="rounded-2xl border border-red-400/40 bg-red-500/10 px-6 py-4 text-left text-sm text-red-200">
          {error}
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap gap-3 text-xs text-white/60">
          {benefits.map(({ icon: Icon, label }) => (
            <span key={label} className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2">
              <Icon className="size-4 text-neon" /> {label}
            </span>
          ))}
        </div>
        <div className="flex gap-3">
          <button
            className="rounded-full border border-white/10 px-4 py-2 text-sm font-semibold text-white/80 transition hover:border-white/30 hover:bg-white/10"
            type="button"
            onClick={clearSelection}
            disabled={isUploading}
          >
            Reset
          </button>
          <button
            className="rounded-full bg-gradient-to-r from-neon via-sapphire to-blush px-6 py-2 text-sm font-semibold text-white shadow-glass transition hover:shadow-[0_22px_60px_rgba(56,189,248,0.35)] disabled:cursor-not-allowed disabled:opacity-50"
            type="button"
            onClick={upload}
            disabled={!file || isUploading}
          >
            {isUploading ? 'Uploading…' : 'Start upload'}
          </button>
        </div>
      </div>
    </section>
  );
}
