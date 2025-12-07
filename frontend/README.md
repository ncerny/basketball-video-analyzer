# Basketball Video Analyzer - Frontend

React + TypeScript frontend application for the Basketball Video Analyzer.

## Tech Stack

- **Framework**: React 19+
- **Language**: TypeScript
- **Build Tool**: Vite 7+
- **Routing**: React Router DOM 7
- **State Management**: Zustand
- **Styling**: Tailwind CSS 4
- **HTTP Client**: Axios
- **Testing**: Vitest + React Testing Library
- **Package Manager**: pnpm

## Setup

### Prerequisites

- Node.js 18+ or Node.js 20+
- pnpm ([installation guide](https://pnpm.io/installation))

### Installation

```bash
cd frontend
pnpm install
```

### Environment Configuration

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` to set your API URL (defaults to `http://localhost:8000`).

### Running the Development Server

```bash
pnpm dev
```

The application will be available at http://localhost:5173

## Development

### Available Scripts

```bash
# Start development server
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview

# Run tests
pnpm test

# Run tests with UI
pnpm test:ui

# Run tests with coverage
pnpm test:coverage

# Lint code
pnpm lint
```

### Project Structure

```
frontend/
├── src/
│   ├── components/      # Reusable React components
│   ├── pages/           # Page-level components
│   ├── hooks/           # Custom React hooks
│   ├── store/           # Zustand state management stores
│   ├── api/             # API client and endpoint functions
│   ├── types/           # TypeScript type definitions
│   ├── utils/           # Utility functions
│   └── test/            # Test setup and utilities
├── public/              # Static assets
├── index.html           # HTML entry point
├── package.json         # Dependencies and scripts
├── vite.config.ts       # Vite configuration
├── vitest.config.ts     # Vitest testing configuration
├── tailwind.config.js   # Tailwind CSS configuration
├── tsconfig.json        # TypeScript configuration
└── README.md
```

## Building for Production

```bash
pnpm build
```

The production build will be in the `dist/` directory.

## Testing

This project uses Vitest and React Testing Library for testing.

### Running Tests

```bash
# Run tests in watch mode
pnpm test

# Run tests once
pnpm test run

# Run with coverage
pnpm test:coverage

# Open test UI
pnpm test:ui
```

### Writing Tests

Place test files next to the components they test with `.test.tsx` or `.test.ts` extension.

Example:
```
src/components/VideoPlayer/
├── VideoPlayer.tsx
└── VideoPlayer.test.tsx
```

## Code Style

- Use TypeScript for type safety
- Follow React best practices and hooks patterns
- Use Tailwind CSS utility classes for styling
- Keep components small and focused
- Write tests for components and critical logic

## Environment Variables

- `VITE_API_URL`: Backend API base URL (default: `http://localhost:8000`)

All environment variables must be prefixed with `VITE_` to be accessible in the application.
