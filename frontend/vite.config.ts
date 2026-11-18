import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Only the API is proxied; the backend serves everything under /api.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // Keep SSE responses (job events, agent chat) unbuffered.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            const type = proxyRes.headers['content-type'] ?? ''
            if (type.includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache, no-transform'
              proxyRes.headers['x-accel-buffering'] = 'no'
              delete proxyRes.headers['content-length']
            }
          })
        },
      },
    },
  },
})
