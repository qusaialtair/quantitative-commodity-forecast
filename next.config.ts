import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // ── Standalone output ────────────────────────────────────────────────────
  // Produces .next/standalone — a self-contained Node.js server with only
  // the files it actually uses, no node_modules tree. Critical for the
  // Oracle Cloud AMD Micro container: cuts the runtime image to ~150MB.
  output: "standalone",

  typescript: {
    // !! WARN !!
    // Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    // !! WARN !!
    ignoreBuildErrors: true,
  },
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },

  // ── Backend proxy rewrite ────────────────────────────────────────────────
  // In production the QCTF_API_PROXY_TARGET env var points at the FastAPI
  // container's internal Docker network address (http://backend:8000).
  // Locally it falls back to http://localhost:8000.
  async rewrites() {
    const backend =
      process.env.QCTF_API_PROXY_TARGET ?? "http://localhost:8000";
    return [
      {
        source: "/qctf-backend/:path*",
        destination: `${backend}/:path*`,
      },
    ];
  },
};

export default nextConfig;
