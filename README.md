# NovaShare

NovaShare is a premium-styled large file sharing experience that lets you upload and distribute transfers up to **10 GB** with shareable download links. It pairs a glassmorphism React + Tailwind interface with a performant Express API, streaming uploads straight to disk and keeping metadata in SQLite.

## Features

- 🚀 **Massive transfers** – upload a single file up to 10 GB with real-time progress.
- 💎 **Premium UI** – gradient-rich, glassy interface tailored for client deliveries.
- 🔐 **Secure by default** – AES-ready storage design, unique share codes, optional deletion.
- 📊 **Instant insights** – track number of files and total storage consumed.
- 🔗 **Shareable links** – one-click copy links for every upload, backed by streaming downloads.

## Project structure

```
.
├── client/        # React + Vite + Tailwind front-end
├── server/        # Express API with SQLite metadata store
├── uploads/       # Runtime file storage (created automatically)
└── data/          # SQLite database location (auto-created)
```

## Prerequisites

- Node.js 18+
- npm 9+

## Setup

Install dependencies from the repository root:

```bash
npm install
```

Create an environment file for the server (optional unless overriding defaults):

```bash
cp server/.env.example server/.env
```

Adjust the values if you plan to expose the API publicly.

## Running locally

Start both the API and the front-end in parallel:

```bash
npm run dev
```

- Front-end: http://localhost:5173
- API: http://localhost:4000

The React client reads the API origin from `VITE_API_URL`. You can override it by creating a `client/.env` file:

```
VITE_API_URL=http://localhost:4000
```

## Production build

Generate the static assets:

```bash
npm run build
```

Deploy the `client/dist` folder behind any static host and run the API with:

```bash
npm run start
```

Ensure the environment variable `CLIENT_ORIGIN` is set to the domain serving the front-end.

## Testing

Execute the automated tests (covers utility code and upload flow):

```bash
npm test
```

## API overview

| Method | Endpoint                | Description                               |
|--------|-------------------------|-------------------------------------------|
| GET    | `/health`               | Health check                               |
| GET    | `/api/files`            | List recent uploads                        |
| POST   | `/api/files`            | Upload a file (field name: `file`)         |
| GET    | `/api/files/:code`      | Metadata for a specific upload             |
| GET    | `/api/files/:code/download` | Download the file associated with `code` |
| DELETE | `/api/files/:code`      | Remove a file and its metadata             |

## Security notes

- Express serves downloads via streaming to avoid memory bloat. Files sit on disk inside `/uploads` with hashed filenames.
- Share codes exclude ambiguous characters for easier verbal sharing.
- Multer enforces a 10 GB hard cap via `LIMIT_FILE_SIZE`.
- SQLite is initialized in Write-Ahead Logging (WAL) mode for consistency.

## Next steps

- Add authentication/teams for multi-user vaults.
- Wire encryption at rest by wrapping the upload stream.
- Enable resumable uploads with chunk orchestration for unreliable networks.
- Plug into cloud storage (S3, Azure Blob, etc.) behind the same interface.
