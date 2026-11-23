"use client";

import { useEffect, useRef } from "react";
import { TrainingExamples } from "../components/explainer/TrainingExamples";
import { ArchitectureOverview } from "../components/explainer/ArchitectureOverview";
import { ModelWalkthrough } from "../components/explainer/ModelWalkthrough";
import { PaperLink } from "../components/explainer/MathText";

export default function Home() {
  const progressRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const update = () => {
      const progress = window.scrollY / Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      progressRef.current?.style.setProperty("transform", `scaleX(${progress})`);
    };
    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, []);
  return <>
    <a className="skip" href="#training-data">Skip to the lesson</a>
    <header className="site-header">
      <a className="brand" href="#top">LiDDA</a>
      <nav aria-label="Lesson sections"><a href="#training-data">Data</a><a href="#architecture-check">Model</a><a href="#attention-readout">Credit</a><a href="#notes">Paper</a></nav>
      <div className="reading-progress" ref={progressRef} aria-hidden="true" />
    </header>
    <main id="top">
      <section className="hero" aria-labelledby="page-title">
        <h1 id="page-title">LiDDA</h1>
        <p className="hero-name">LinkedIn Data-Driven Attribution</p>
        <p className="hero-subtitle">An application of attention · POLS 4728</p>
      </section>
      <TrainingExamples />
      <ArchitectureOverview />
      <ModelWalkthrough />
      <section className="source-note" id="notes" aria-labelledby="source-title">
        <h2 id="source-title">Paper and illustration</h2>
        <p><PaperLink section="S2">LiDDA, §§2.1–2.4 and Figure 1</PaperLink> supplies the architecture. The numbers and two heads are illustrative; the paper does not report its head count. For attribution, we sum incoming attention over valid query positions and heads, then normalize across touches. The paper does not fully specify the query-axis reduction, so this is an explicit teaching convention.</p>
      </section>
    </main>
    <footer className="site-footer"><span>POLS 4728 · Attention applications</span><a href="#top">Back to top ↑</a></footer>
  </>;
}
