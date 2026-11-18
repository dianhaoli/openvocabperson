# Frontend - React + TypeScript + Vite

This is the React-based frontend for the Human-Centric Vision Analysis application.

## Development

```bash
# Install dependencies
npm install

# Start development server (with API proxy to localhost:8000)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
src/
├── api/client.ts      # Typed API client (docs/API.md), ApiError, SSE-over-fetch generator
├── types/             # Types mirroring docs/API.md (snake_case JSON fields)
├── context/           # App state (reducer + context), toasts, health polling
├── hooks/             # useApi, useJobTracker (job SSE), useWorkspaceJob, useObjectUrl, useElementSize
├── components/
│   ├── layout/        # Header (health + capabilities), Sidebar tabs
│   ├── analyze/       # Upload + open-vocab query + options + plan preview
│   ├── agent/         # Agent chat (SSE), tool-call cards with galleries
│   ├── search/        # Text / image search
│   ├── library/       # Media library, scan job, recent jobs
│   ├── people/        # Persons grid, review queue, find-by-photo, person modal
│   ├── workspace/     # Image canvas / video overlay + timeline, legend, entity cards, scene Q&A
│   ├── detail/        # Selected-entity panel (identity, attributes, Q&A)
│   ├── models/        # Models & capabilities modal
│   ├── jobs/          # Job progress panel, plan view
│   ├── common/ ui/    # Shared building blocks
└── utils/             # colors, geometry, formatting, canvas drawing
```

The dev server proxies only `/api` to `http://localhost:8000`.

## Docker

The frontend can be built as a Docker container:

```bash
docker build -t vision-frontend .
docker run -p 3000:80 vision-frontend
```

Or with docker-compose from the parent directory:

```bash
docker-compose up frontend
```

## Tech Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS v4** - Styling
- **PostCSS** - CSS processing
