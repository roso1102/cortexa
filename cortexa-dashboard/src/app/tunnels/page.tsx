import { getTunnels } from "@/lib/cortexaApi";

export const dynamic = "force-dynamic";

export default async function TunnelsPage() {
  const { tunnels } = await getTunnels();

  // Deduplicate by id if present
  const seen = new Set<string>();
  const items = tunnels.filter((t) => {
    const id = (t.id as string) || "";
    if (!id) return true;
    if (seen.has(id)) return false;
    seen.add(id);
    return true;
  });

  return (
    <div className="space-y-6">
      <header className="rounded-2xl border border-zinc-200 bg-white p-6">
        <div className="text-sm text-zinc-500">Tunnels</div>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">Your semantic tunnels</h1>
        <p className="mt-2 text-sm text-zinc-600">
          Tunnels are weekly theme clusters (currently tag-based) named by the LLM.
        </p>
      </header>

      <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {items.map((t, idx) => {
          const name = (t.tunnel_name as string) || "Untitled Tunnel";
          const count = t.memory_count;
          const core = (t.core_tag as string) || "";
          const raw = (t.raw_content as string) || "";
          return (
            <div key={(t.id as string) || `${idx}`} className="rounded-2xl border border-zinc-200 bg-white p-6">
              <div className="flex items-start justify-between gap-3">
                <div className="text-lg font-semibold tracking-tight">{name}</div>
                {typeof count === "number" ? (
                  <span className="rounded-full bg-zinc-100 px-2 py-1 text-xs text-zinc-700">{count} memories</span>
                ) : null}
              </div>
              {core ? <div className="mt-2 text-sm text-zinc-600">Core tag: {core}</div> : null}
              {raw ? (
                <pre className="mt-4 whitespace-pre-wrap text-sm leading-6 text-zinc-700">{raw.slice(0, 600)}</pre>
              ) : null}
            </div>
          );
        })}
        {!items.length ? <div className="text-sm text-zinc-600">No tunnels yet.</div> : null}
      </section>
    </div>
  );
}

