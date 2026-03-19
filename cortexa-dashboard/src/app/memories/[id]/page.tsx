import { getMemoryById } from "@/lib/cortexaApi";

export const dynamic = "force-dynamic";

export default async function MemoryDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const data = await getMemoryById(id);
  const m = data.item;

  const title = (m.title as string) || (m.file_name as string) || (m.url as string) || "Memory";
  const st = (m.source_type as string) || "text";
  const raw = (m.raw_content as string) || "";
  const tagsArr = Array.isArray(m.tags) ? (m.tags as string[]) : [];

  return (
    <div className="space-y-6">
      <header className="rounded-2xl border border-zinc-200 bg-white p-6">
        <div className="text-sm text-zinc-500">Memory</div>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">{title}</h1>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-zinc-100 px-2 py-1 text-xs text-zinc-700">{st}</span>
          {m.id ? <span className="rounded-full border border-zinc-200 px-2 py-1 text-xs text-zinc-600">id: {m.id as string}</span> : null}
        </div>
        {m.url ? (
          <a className="mt-3 block break-all text-sm text-blue-700 hover:underline" href={m.url as string} target="_blank" rel="noreferrer">
            {m.url as string}
          </a>
        ) : null}
      </header>

      {tagsArr.length ? (
        <section className="rounded-2xl border border-zinc-200 bg-white p-6">
          <div className="text-sm font-medium">Tags</div>
          <div className="mt-3 flex flex-wrap gap-2">
            {tagsArr.map((t) => (
              <span key={t} className="rounded-full border border-zinc-200 px-2 py-1 text-xs text-zinc-600">
                {t}
              </span>
            ))}
          </div>
        </section>
      ) : null}

      <section className="rounded-2xl border border-zinc-200 bg-white p-6">
        <div className="text-sm font-medium">Full text</div>
        <pre className="mt-3 whitespace-pre-wrap text-sm leading-6 text-zinc-800">{raw}</pre>
      </section>
    </div>
  );
}

