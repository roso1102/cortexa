import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "cortexa dashboard",
  description: "Interactive dashboard for your cortexa memory system",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-zinc-50 text-zinc-900`}
      >
        <div className="min-h-screen">
          <div className="mx-auto flex w-full max-w-6xl gap-6 px-4 py-6">
            <aside className="hidden w-56 flex-shrink-0 md:block">
              <div className="rounded-2xl border border-zinc-200 bg-white p-4">
                <div className="text-lg font-semibold tracking-tight">cortexa</div>
                <div className="mt-1 text-xs text-zinc-500">Memory dashboard</div>
                <nav className="mt-4 flex flex-col gap-1 text-sm">
                  <a className="rounded-lg px-3 py-2 hover:bg-zinc-50" href="/">Dashboard</a>
                  <a className="rounded-lg px-3 py-2 hover:bg-zinc-50" href="/memories">Memories</a>
                  <a className="rounded-lg px-3 py-2 hover:bg-zinc-50" href="/tunnels">Tunnels</a>
                  <a className="rounded-lg px-3 py-2 hover:bg-zinc-50" href="/profile">Profile</a>
                </nav>
              </div>
            </aside>
            <main className="flex-1">{children}</main>
          </div>
          <footer className="mx-auto w-full max-w-6xl px-4 pb-8 text-xs text-zinc-500">
            Powered by Koyeb + Pinecone + Gemini + Groq/OpenRouter.
          </footer>
        </div>
      </body>
    </html>
  );
}
