import "../app/globals.css";
import { useEffect } from "react";
import { createRoot } from "react-dom/client";
import Home from "../app/page";

function StaticLesson() {
  useEffect(() => {
    // The initial anchor can precede React's first render on a static host.
    // Resolve it after the lesson mounts so shared section links still work.
    if (!window.location.hash) return;
    let id: string;
    try {
      id = decodeURIComponent(window.location.hash.slice(1));
    } catch {
      return;
    }
    const frame = window.requestAnimationFrame(() => {
      document.getElementById(id)?.scrollIntoView({ block: "start", behavior: "instant" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, []);

  return <Home />;
}

const root = document.getElementById("root");
if (!root) throw new Error("The static lesson is missing its root element.");
createRoot(root).render(<StaticLesson />);
