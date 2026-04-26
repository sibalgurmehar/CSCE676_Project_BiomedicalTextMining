import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#07111f",
        mist: "#edf4f5",
        sea: "#0e7490",
        coral: "#e76f51",
        sand: "#f6e7cf"
      },
      fontFamily: {
        sans: ["var(--font-sans)"],
        display: ["var(--font-display)"]
      },
      boxShadow: {
        soft: "0 20px 60px rgba(7, 17, 31, 0.12)"
      },
      backgroundImage: {
        grid: "radial-gradient(circle at 1px 1px, rgba(7, 17, 31, 0.08) 1px, transparent 0)"
      }
    }
  },
  plugins: []
};

export default config;

