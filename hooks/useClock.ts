"use client";

import { useEffect, useState } from "react";

export function useClock() {
  const [time, setTime] = useState("");

  useEffect(() => {
    const tick = () => {
      const now = new Date();
      const formatted = now.toISOString().replace("T", "  ").slice(0, 19) + " UTC";
      setTime(formatted);
    };

    const frame = requestAnimationFrame(tick);
    const id = setInterval(tick, 1000);
    return () => {
      cancelAnimationFrame(frame);
      clearInterval(id);
    };
  }, []);

  return time;
}
