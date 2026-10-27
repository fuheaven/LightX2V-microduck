import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    proxy: {
      "/api": "http://127.0.0.1:8771",
      "/media": "http://127.0.0.1:8771",
    },
  },
  build: { outDir: "dist" },
});
