import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = fileURLToPath(new URL("../", import.meta.url));
const repositoryRoot = resolve(packageRoot, "../..");
const output = resolve(packageRoot, "dist-static");
const destination = resolve(repositoryRoot, "docs/lidda");

if (resolve(repositoryRoot, "lesson-sources/lidda") !== resolve(packageRoot)) {
  throw new Error("Run this source package from the repository's lesson-sources/lidda directory.");
}

// Rebuild before replacing the generated page, so an old bundle is not staged.
await import("./export-static.mjs");
await mkdir(dirname(destination), { recursive: true });
await rm(destination, { recursive: true, force: true });
await cp(output, destination, { recursive: true });
console.log(`Updated generated page: ${destination}`);
console.log("This copied local files only. Review and commit them through the course repository workflow.");
