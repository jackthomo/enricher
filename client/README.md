This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, start the FastAPI backend from the `enrich` folder:

```bash
uvicorn enrichment_agent.api:app --host 0.0.0.0 --port 8000 --reload
```

Then, in this `client` folder, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser. The UI will POST to the FastAPI endpoint at `http://localhost:8000/enrich` by default. To point to a different backend, set `NEXT_PUBLIC_API_BASE` in `.env.local`.

Use the **Check connection** button in the header to hit `/health` before sending full enrichment requests. You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

After each request the right-hand panel shows both the structured `info` and a Trace section listing the agent/tool calls returned by the backend. Use the **Pause/Cancel** button during a run to abort an in-flight request.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
