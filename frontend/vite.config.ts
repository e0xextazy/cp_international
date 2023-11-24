import { defineConfig } from 'vite';

import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import dotenv from 'dotenv';

dotenv.config();

export default defineConfig({
  plugins: [
    react({
      include: '**/*.tsx',
    }),
  ],
  css: {
    preprocessorOptions: {
      scss: {
        additionalData: `@import "src/variables.scss";`,
      },
    },
  },
  server: {
    watch: {
      usePolling: true,
    },
  },
  resolve: {
    alias: {
      '@fonts': resolve('./public/fonts'),
    },
  },
});
