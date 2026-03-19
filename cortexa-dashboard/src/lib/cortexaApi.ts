export type MemoryItem = Record<string, unknown> & {
  id?: string;
  source_type?: string;
  title?: string;
  url?: string;
  file_name?: string;
  raw_content?: string;
  created_at?: string;
  created_at_ts?: number;
  tags?: string[];
  score?: number;
  priority_score?: number;
  tunnel_name?: string;
  memory_count?: number;
  core_tag?: string;
};

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`Missing required env var: ${name}`);
  return v;
}

function baseUrl(): string {
  const raw = requireEnv("NEXT_PUBLIC_API_URL");
  return raw.replace(/\/+$/, "");
}

async function apiFetch<T>(path: string): Promise<T> {
  const secret = requireEnv("DASHBOARD_SECRET");
  const res = await fetch(`${baseUrl()}${path}`, {
    method: "GET",
    headers: {
      "X-Dashboard-Token": secret,
    },
    cache: "no-store",
  });

  if (!res.ok) {
    // Try to parse JSON error first, fall back to text
    let detail = "";
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      const j = await res.json().catch(() => ({}));
      detail = j.error || JSON.stringify(j);
    } else {
      const t = await res.text().catch(() => "");
      // Strip HTML tags from Flask error pages
      detail = t.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 300);
    }
    throw new Error(`API ${path} → ${res.status}: ${detail}`);
  }

  return (await res.json()) as T;
}

export async function getTodaySummary(): Promise<{ text: string; generated_at: string }> {
  return apiFetch("/api/summary/today");
}

export async function getProfile(): Promise<{ snapshot: string }> {
  return apiFetch("/api/profile");
}

export async function getTunnels(): Promise<{ tunnels: MemoryItem[] }> {
  return apiFetch("/api/tunnels");
}

export async function getMemories(params?: {
  q?: string;
  source_type?: string;
  tag?: string;
  page?: number;
  per_page?: number;
}): Promise<{ items: MemoryItem[]; total: number; page: number; per_page: number }> {
  const sp = new URLSearchParams();
  if (params?.q) sp.set("q", params.q);
  if (params?.source_type) sp.set("source_type", params.source_type);
  if (params?.tag) sp.set("tag", params.tag);
  if (params?.page) sp.set("page", String(params.page));
  if (params?.per_page) sp.set("per_page", String(params.per_page));
  const qs = sp.toString();
  return apiFetch(`/api/memories${qs ? `?${qs}` : ""}`);
}

export async function getMemoryById(id: string): Promise<{ item: MemoryItem }> {
  return apiFetch(`/api/memories/${encodeURIComponent(id)}`);
}

