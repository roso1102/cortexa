import { getTodaySummary } from "@/lib/cortexaApi";

export const dynamic = "force-dynamic";

export default function Home() {
  // Server component: runs on Vercel, keeps DASHBOARD_SECRET private
  return (
    <Dashboard />
  );
}

async function Dashboard() {
  let summary = "";
  let error: string | null = null;
  try {
    const res = await getTodaySummary();
    summary = res.text;
  } catch (e) {
    error = e instanceof Error ? e.message : "Unknown error";
  }

  return (
    <div className="space-y-6">
      <header className="rounded-2xl border border-zinc-200 bg-white p-6">
        <div className="text-sm text-zinc-500">Dashboard</div>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">Today in your memory</h1>
        <p className="mt-2 text-sm text-zinc-600">
          This pulls from your live Koyeb API. If it fails, check <code className="rounded bg-zinc-50 px-1">NEXT_PUBLIC_API_URL</code> and{" "}
          <code className="rounded bg-zinc-50 px-1">DASHBOARD_SECRET</code>.
        </p>
      </header>

      <section className="rounded-2xl border border-zinc-200 bg-white p-6">
        <div className="text-sm font-medium">Summary</div>
        {error ? (
          <div className="mt-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        ) : (
          <pre className="mt-3 whitespace-pre-wrap text-sm leading-6 text-zinc-800">{summary || "No summary yet."}</pre>
        )}
      </section>
    </div>
  );
}
