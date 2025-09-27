const UNITS = ['B', 'KB', 'MB', 'GB', 'TB'];

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const exponent = Math.min(Math.floor(Math.log10(bytes) / 3), UNITS.length - 1);
  const size = bytes / 10 ** (exponent * 3);
  return `${size.toFixed(size >= 10 || size % 1 === 0 ? 0 : 1)} ${UNITS[exponent]}`;
}

export function formatRelativeDate(value) {
  if (!value) return 'N/A';
  const date = typeof value === 'string' ? new Date(value) : value;
  const now = new Date();
  const diffMs = now - date;
  const diffMinutes = Math.round(diffMs / (1000 * 60));
  if (diffMinutes < 1) return 'just now';
  if (diffMinutes < 60) return `${diffMinutes} min ago`;
  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours} hr ago`;
  const diffDays = Math.round(diffHours / 24);
  if (diffDays < 30) return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`;
  return date.toLocaleDateString();
}
