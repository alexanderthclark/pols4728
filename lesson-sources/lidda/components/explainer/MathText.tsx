import katex from "katex";
import { memo, type ReactNode } from "react";

export const PAPER_URL = "https://arxiv.org/html/2505.09861v3";

export const MathText = memo(function MathText({ children, block = false }: { children: string; block?: boolean }) {
  return <span className={block ? "math-block" : "math-inline"} dangerouslySetInnerHTML={{ __html: katex.renderToString(children, { displayMode: block, throwOnError: true, output: "htmlAndMathml" }) }} />;
});

export function PaperLink({ section, children }: { section: string; children?: ReactNode }) {
  return <a className="paper-link" href={`${PAPER_URL}#${section}`} target="_blank" rel="noreferrer">{children || "Read the paper"} ↗</a>;
}
