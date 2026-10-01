import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// O painel lê os resultados do orquestrador direto do disco, servidos pelo
// http.server na 5174. O proxy evita CORS e mantém os caminhos relativos.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    proxy: { '/results': { target: 'http://127.0.0.1:5174', changeOrigin: true } },
  },
})
