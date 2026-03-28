import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        ws: true,
      },
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
      },
    },
  },
  build: {
    sourcemap: false,
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) {
            return;
          }

          if (id.includes("echarts") || id.includes("echarts-for-react")) {
            return "charts-echarts";
          }

          if (id.includes("recharts")) {
            return "charts-recharts";
          }

          if (id.includes("reactflow")) {
            return "workflow-reactflow";
          }

          if (id.includes("antd") || id.includes("@ant-design") || id.includes("rc-")) {
            return "vendor-antd";
          }
        },
      },
    },
  },
  test: {
    environment: "node",
  },
});
