/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", "Inter", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "JetBrains Mono", "ui-monospace", "monospace"],
      },
      colors: {
        ebony: {
          DEFAULT: "#09090b",
          alt: "#0c0c0e",
          950: "#050506",
        },
        charcoal: {
          DEFAULT: "#141416",
          light: "#1c1c1f",
          dark: "#0c0c0e",
        },
        surface: {
          DEFAULT: "#18181b",
          raised: "#1f1f23",
          overlay: "#27272a",
        },
        border: {
          DEFAULT: "#27272a",
          subtle: "#1f1f23",
          strong: "#3f3f46",
          focus: "#52525b",
        },
        text: {
          primary: "#fafafa",
          secondary: "#a1a1aa",
          muted: "#71717a",
        },
        positive: "#22c55e",
        negative: "#ef4444",
        warning: "#f59e0b",
      },
    },
  },
  plugins: [],
};
