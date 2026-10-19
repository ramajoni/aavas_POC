import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  build: {
    outDir: "dist/client",
    // Two surfaces, one build: the lending workspace at / and the OS console at
    // /os.html. They share the component library and the API client; only the
    // shell differs.
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        os: resolve(__dirname, "os.html"),
      },
    },
  },
  optimizeDeps: {
    include: ["react", "react-dom/client"],
  },
  server: {
    host: "0.0.0.0",
    allowedHosts: ["terminal.local"],
    proxy: { "/api": { target: process.env.AAVAS_API_URL || "http://127.0.0.1:8000", changeOrigin: true, rewrite: path => path.replace(/^\/api/, "") } },
    warmup: {
      clientFiles: ["./src/main.jsx"],
    },
  },
  plugins: [react()],
});
