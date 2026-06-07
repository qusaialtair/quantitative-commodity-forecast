import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { DeploymentProvider } from "@/components/providers/DeploymentProvider";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Altair MK1",
  description: "Institutional quantitative trading terminal — demo deployment",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} min-h-screen bg-ebony font-sans text-text-primary antialiased`}
        suppressHydrationWarning
      >
        <DeploymentProvider>{children}</DeploymentProvider>
      </body>
    </html>
  );
}
