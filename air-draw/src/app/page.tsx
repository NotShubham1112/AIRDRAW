"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, ArrowRight, MousePointer2, Sparkles, Eraser, Move, Github, BookOpen } from "lucide-react";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function LandingPage() {
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const handleStart = () => {
    setIsLoading(true);
    router.push("/draw");
  };

  return (
    <div className="relative min-h-screen w-screen overflow-hidden bg-[#0a0a0a] text-zinc-100 selection:bg-zinc-500/30">
      {/* Subtle Background Gradient */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(40,40,40,0.4),transparent_70%)]" />
      </div>

      {/* Top Navigation */}
      <nav className="absolute top-0 right-0 z-50 flex w-full items-center justify-end gap-3 p-6 sm:p-8">
        <Link 
          href="https://github.com/NotShubham1112/AIRDRAW" 
          target="_blank" 
          rel="noopener noreferrer"
          className="flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/50 px-4 py-2 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-700 hover:text-white backdrop-blur-md"
        >
          <BookOpen className="h-4 w-4" />
          <span>Docs</span>
        </Link>
        <Link 
          href="https://github.com/NotShubham1112/AIRDRAW" 
          target="_blank" 
          rel="noopener noreferrer"
          className="flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/50 px-4 py-2 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-700 hover:text-white backdrop-blur-md"
        >
          <Github className="h-4 w-4" />
          <span>GitHub</span>
        </Link>
      </nav>

      <main className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6 py-20 text-center">
        {/* Top Badge */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="mb-8 inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/50 px-4 py-1.5 backdrop-blur-md"
        >
          <div className="h-1.5 w-1.5 rounded-full bg-zinc-400 animate-pulse" />
          <span className="text-xs font-medium tracking-wider text-zinc-400 uppercase">
            Gesture Canvas 2.0
          </span>
        </motion.div>

        {/* Main Heading */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1, ease: "easeOut" }}
          className="max-w-4xl space-y-4"
        >
          <h1 className="text-5xl font-bold tracking-tight sm:text-7xl">
            Draw in the air,{" "}
            <span className="bg-gradient-to-r from-zinc-200 to-zinc-500 bg-clip-text text-transparent italic">
              effortlessly
            </span>
          </h1>
          <p className="mx-auto max-w-2xl text-lg text-zinc-400 sm:text-xl leading-relaxed">
            Experience the future of sketching. Use simple hand gestures to create art directly in your browser. No hardware required, just your vision.
          </p>
        </motion.div>

        {/* Action Button */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.3, ease: "easeOut" }}
          className="mt-12"
        >
          <button
            onClick={handleStart}
            disabled={isLoading}
            className="group relative flex items-center gap-3 overflow-hidden rounded-full bg-zinc-100 px-8 py-4 font-semibold text-zinc-900 transition-all hover:bg-white hover:shadow-[0_0_30px_rgba(255,255,255,0.2)] disabled:opacity-80"
          >
            <AnimatePresence mode="wait">
              {isLoading ? (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 1.2 }}
                  className="flex items-center gap-2"
                >
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Preparing Canvas...</span>
                </motion.div>
              ) : (
                <motion.div
                  key="default"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex items-center gap-2"
                >
                  <span>Start drawing</span>
                  <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
                </motion.div>
              )}
            </AnimatePresence>
          </button>
        </motion.div>

        {/* Features Preview */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mt-24 grid grid-cols-2 gap-8 md:grid-cols-4"
        >
          <FeatureIcon icon={<Sparkles className="h-5 w-5" />} label="Draw" />
          <FeatureIcon icon={<Move className="h-5 w-5" />} label="Move" />
          <FeatureIcon icon={<Eraser className="h-5 w-5" />} label="Erase" />
          <FeatureIcon icon={<MousePointer2 className="h-5 w-5" />} label="Pinch" />
        </motion.div>
      </main>

      {/* Footer Hint */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 1 }}
        className="fixed bottom-6 w-full flex flex-col items-center gap-2 text-center"
      >
        <span className="text-[11px] font-medium tracking-widest text-zinc-600 uppercase">
          Best with a desktop camera
        </span>
        <span className="text-xs font-medium text-zinc-500">
          Made by <a href="https://github.com/shubham-kambli" target="_blank" rel="noopener noreferrer" className="text-zinc-300 hover:text-white transition-colors">Shubham Kambli</a>
        </span>
      </motion.div>
    </div>
  );
}

function FeatureIcon({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="flex flex-col items-center gap-3 group">
      <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-zinc-800 bg-zinc-900/50 text-zinc-400 transition-all group-hover:border-zinc-700 group-hover:text-zinc-200">
        {icon}
      </div>
      <span className="text-xs font-semibold tracking-wider text-zinc-500 uppercase group-hover:text-zinc-400 line-clamp-1">
        {label}
      </span>
    </div>
  );
}
