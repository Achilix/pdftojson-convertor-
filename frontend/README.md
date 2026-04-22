# Frontend (Next.js)

This frontend is connected to the Python legal search API in the same repository.

## Prerequisites

- Node.js 20+
- Python API available in the repository root

## Configuration

Create a local env file:

```bash
cp .env.local.example .env.local
```

Default API URL in [.env.local.example](.env.local.example):

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Run Locally

1. Start the Python API from repository root:

```bash
python src/api.py
```

2. Start Next.js frontend from [frontend](.):

```bash
npm run dev
```

3. Open:

```text
http://localhost:3000
```

## Scripts

- `npm run dev` - start development server
- `npm run build` - production build
- `npm run start` - run built app
- `npm run lint` - run ESLint

## Main Files

- [src/app/page.tsx](src/app/page.tsx) - search UI and API calls
- [src/app/globals.css](src/app/globals.css) - styles
- [.env.local.example](.env.local.example) - API base URL template
