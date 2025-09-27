/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        midnight: '#0f172a',
        sapphire: '#1d4ed8',
        neon: '#38bdf8',
        blush: '#ec4899'
      },
      fontFamily: {
        display: ['"Plus Jakarta Sans"', 'sans-serif'],
        body: ['"Inter"', 'sans-serif']
      },
      boxShadow: {
        glass: '0 18px 45px rgba(56, 189, 248, 0.25)'
      }
    }
  },
  plugins: []
};
