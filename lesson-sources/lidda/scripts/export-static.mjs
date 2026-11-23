import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { build } from "vite";
import react from "@vitejs/plugin-react";

const projectRoot = fileURLToPath(new URL("../", import.meta.url));

// This independent entry bundles the React lesson without loading the Sites,
// Next, Vinext, or Worker configuration. Relative assets allow deployment at
// /pols4728/lidda/ or any other static-host subdirectory.
await build({
  configFile: false,
  root: resolve(projectRoot, "static"),
  publicDir: resolve(projectRoot, "public"),
  base: "./",
  plugins: [react()],
  css: {
    // The lesson uses ordinary CSS. Do not load the framework's PostCSS config.
    postcss: { plugins: [] },
  },
  build: {
    outDir: resolve(projectRoot, "dist-static"),
    emptyOutDir: true,
    target: "es2020",
    assetsInlineLimit: 0,
    sourcemap: false,
  },
});
