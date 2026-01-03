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
├── api/            # Typed API client functions
├── components/
│   ├── canvas/     # Canvas with bounding boxes
│   ├── entity/     # Entity panel and Q&A
│   ├── layout/     # Header, Sidebar, MainLayout
│   ├── results/    # Results grid and cards
│   ├── sidebar/    # Upload, Search, History tabs
│   └── ui/         # Reusable UI components
├── context/        # React Context for global state
├── hooks/          # Custom React hooks
├── types/          # TypeScript type definitions
└── utils/          # Utility functions
```

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
