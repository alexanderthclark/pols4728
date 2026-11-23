/** The four synthetic events shared by every numerical lesson. */
export const TOUCHES = [
  { id: "display", label: "Display ad", shortLabel: "Display", action: "Impression", day: -21, color: "#234E70" },
  { id: "social", label: "Paid social", shortLabel: "Social", action: "Click", day: -14, color: "#A54F32" },
  { id: "video", label: "Video ad", shortLabel: "Video", action: "View", day: -7, color: "#007F73" },
  { id: "search", label: "Paid search", shortLabel: "Search", action: "Click", day: -1, color: "#7B4F9D" },
] as const;
