import { getMemories } from "@/lib/cortexaApi";

export const dynamic = "force-dynamic";

export default async function MemoriesPage({
  searchParams,
}: {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = (await searchParams) || {};
  const q = typeof sp.q === "string" ? sp.q : "";
  const source_type = typeof sp.source_type === "string" ? sp.source_type : "";
  const tag = typeof sp.tag === "string" ? sp.tag : "";
  const page = typeof sp.page === "string" ? Math.max(parseInt(sp.page, 10) || 1, 1) : 1;

  const data = await getMemories({ q, source_type, tag, page, per_page: 20 });

  return (
    <div className="space-y-6">
      <header className="rounded-2xl border border-zinc-200 bg-white p-6">
        <div className="text-sm text-zinc-500">Memories</div>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">Browse your memory store</h1>
        <p className="mt-2 text-sm text-zinc-600">Filter by source type or tags. Search is semantic (best-effort).</p>
      </header>

      <form className="rounded-2xl border border-zinc-200 bg-white p-6">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
          <input
            name="q"
            defaultValue={q}
            placeholder="Search (semantic)..."
            className="h-10 rounded-xl border border-zinc-200 px-3 text-sm outline-none focus:border-zinc-400"
          />
          <input
            name="source_type"
            defaultValue={source_type}
            placeholder="source_type (text/link/pdf/...)"
            className="h-10 rounded-xl border border-zinc-200 px-3 text-sm outline-none focus:border-zinc-400"
          />
          <input
            name="tag"
            defaultValue={tag}
            placeholder="tag (e.g. fpga)"
            className="h-10 rounded-xl border border-zinc-200 px-3 text-sm outline-none focus:border-zinc-400"
          />
          <button className="h-10 rounded-xl bg-zinc-900 px-3 text-sm font-medium text-white hover:bg-zinc-800">
            Apply
          </button>
        </div>
      </form>

      <section className="rounded-2xl border border-zinc-200 bg-white">
        <div className="border-b border-zinc-200 p-4 text-sm text-zinc-600">
          Showing {data.items.length} of {data.total} (page {data.page})
        </div>
        <div className="divide-y divide-zinc-100">
          {data.items.map((m, idx) => {
            const title = (m.title as string) || (m.file_name as string) || (m.url as string) || "Untitled";
            const st = (m.source_type as string) || "text";
            const raw = (m.raw_content as string) || "";
            const tagsArr = Array.isArray(m.tags) ? (m.tags as string[]) : [];
            return (
              <div key={(m.id as string) || `${idx}`} className="p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full bg-zinc-100 px-2 py-1 text-xs text-zinc-700">{st}</span>
                  {m.id ? (
                    <a className="text-sm font-medium text-blue-700 hover:underline" href={`/memories/${m.id as string}`}>
                      {title}
                    </a>
                  ) : (
                    <div className="text-sm font-medium">{title}</div>
                  )}
                </div>
                {m.url ? (
                  <a className="mt-1 block break-all text-sm text-blue-700 hover:underline" href={m.url as string} target="_blank" rel="noreferrer">
                    {m.url as string}
                  </a>
                ) : null}
                <div className="mt-2 whitespace-pre-wrap text-sm leading-6 text-zinc-700">{raw}</div>
                {tagsArr.length ? (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {tagsArr.slice(0, 8).map((t) => (
                      <span key={t} className="rounded-full border border-zinc-200 px-2 py-1 text-xs text-zinc-600">
                        {t}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            );
          })}
          {!data.items.length ? <div className="p-6 text-sm text-zinc-600">No results.</div> : null}
        </div>
      </section>

      <Pager page={data.page} total={data.total} perPage={data.per_page} q={q} source_type={source_type} tag={tag} />
    </div>
  );
}

function Pager({
  page,
  total,
  perPage,
  q,
  source_type,
  tag,
}: {
  page: number;
  total: number;
  perPage: number;
  q: string;
  source_type: string;
  tag: string;
}) {
  const maxPage = Math.max(1, Math.ceil(total / perPage));
  const prev = Math.max(1, page - 1);
  const next = Math.min(maxPage, page + 1);

  const mk = (p: number) => {
    const sp = new URLSearchParams();
    if (q) sp.set("q", q);
    if (source_type) sp.set("source_type", source_type);
    if (tag) sp.set("tag", tag);
    sp.set("page", String(p));
    return `/memories?${sp.toString()}`;
  };

  return (
    <div className="flex items-center justify-between">
      <a
        className="rounded-xl border border-zinc-200 bg-white px-4 py-2 text-sm hover:bg-zinc-50"
        href={mk(prev)}
        aria-disabled={page <= 1}
      >
        Prev
      </a>
      <div className="text-sm text-zinc-600">
        Page {page} / {maxPage}
      </div>
      <a className="rounded-xl border border-zinc-200 bg-white px-4 py-2 text-sm hover:bg-zinc-50" href={mk(next)}>
        Next
      </a>
    </div>
  );
}

