"use client";

import { useEffect, useRef, useState } from "react";
import { Download, Trash2, Github } from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

type Stroke = {
  pts: { x: number; y: number }[];
  col: string;
  th: number;
};

type GestureName = "DRAW" | "MOVE" | "ERASE" | "PINCH" | "UNKNOWN";

// ─── One-Euro Filter ─────────────────────────────────────────────────────────
// Reduces jitter while keeping low latency.
// Lower minCutoff  → more smoothing (less jitter), slightly more lag.
// Higher beta      → faster response during fast movement.

class OneEuroFilter {
  private prevX = 0;
  private prevY = 0;
  private dx = 0;
  private dy = 0;
  private lastTime = 0;

  constructor(
    private minCutoff = 0.5,   // tuned down for better jitter suppression
    private beta = 0.02,        // slightly higher: snappier fast-motion tracking
    private dCutoff = 1.0
  ) {}

  private alpha(cutoff: number, dt: number) {
    const r = 2 * Math.PI * cutoff * dt;
    return r / (r + 1);
  }

  filter(x: number, y: number, t: number): { x: number; y: number } {
    if (!this.lastTime) {
      this.prevX = x;
      this.prevY = y;
      this.lastTime = t;
      return { x, y };
    }

    const dt = (t - this.lastTime) / 1000;
    if (dt <= 0) return { x: this.prevX, y: this.prevY };

    // Derivative filter
    const ad = this.alpha(this.dCutoff, dt);
    this.dx = ((x - this.prevX) / dt) * ad + this.dx * (1 - ad);
    this.dy = ((y - this.prevY) / dt) * ad + this.dy * (1 - ad);

    // Adaptive cutoff: faster movement → less smoothing (less lag)
    const speed = Math.hypot(this.dx, this.dy);
    const cutoff = this.minCutoff + this.beta * speed;
    const a = this.alpha(cutoff, dt);

    this.prevX = x * a + this.prevX * (1 - a);
    this.prevY = y * a + this.prevY * (1 - a);
    this.lastTime = t;
    return { x: this.prevX, y: this.prevY };
  }

  reset() {
    this.lastTime = 0;
  }
}

// ─── Constants ────────────────────────────────────────────────────────────────

const GESTURE_COLORS: Record<GestureName, string> = {
  DRAW: "#a78bfa",
  MOVE: "#94a3b8",
  ERASE: "#f87171",
  PINCH: "#fbbf24",
  UNKNOWN: "#6b7280",
};

// ─── Component ────────────────────────────────────────────────────────────────

export default function DrawPage() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [gesture, setGesture] = useState<GestureName>("UNKNOWN");
  const [fps, setFps] = useState(0);

  const handleClear = () => window.dispatchEvent(new CustomEvent("clear-canvas"));

  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = `air-draw-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d", { willReadFrequently: false });
    if (!ctx) return;

    // ── Hand landmark topology ──────────────────────────────────────────────
    const FINGER_TIPS = [4, 8, 12, 16, 20];
    const FINGER_PIPS = [3, 6, 10, 14, 18];
    const HAND_CONNECTIONS: [number, number][] = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [5, 9], [9, 10], [10, 11], [11, 12],
      [9, 13], [13, 14], [14, 15], [15, 16],
      [13, 17], [17, 18], [18, 19], [19, 20],
      [0, 17],
    ];
    type Lm = { x: number; y: number; z: number };

    // ── Per-hand state ──────────────────────────────────────────────────────
    interface PerHandState {
      lastHand2d: { x: number; y: number }[] | null;
      pinched: boolean;
      lastGesture: GestureName | null;
      stableCount: number;
      stableGesture: GestureName;
      currentStroke: Stroke | null;
      filter: OneEuroFilter;
      smoothedCursor: { x: number; y: number } | null;
      selectedIdx: number | null;
      pinchPrev: { x: number; y: number } | null;
      lastPoint: { x: number; y: number } | null;
    }

    const createHandState = (): PerHandState => ({
      lastHand2d: null,
      pinched: false,
      lastGesture: null,
      stableCount: 0,
      stableGesture: "UNKNOWN",
      currentStroke: null,
      // Tuned OneEuroFilter: minCutoff=0.5 (more smoothing), beta=0.02 (velocity adaptation)
      filter: new OneEuroFilter(0.5, 0.02, 1.0),
      smoothedCursor: null,
      selectedIdx: null,
      pinchPrev: null,
      lastPoint: null,
    });

    let handStates: PerHandState[] = [createHandState(), createHandState()];
    let strokes: Stroke[] = [];

    // ── Thresholds ──────────────────────────────────────────────────────────
    const PINCH_ON_THRESHOLD  = 0.40;   // tighter pinch detection
    const PINCH_OFF_THRESHOLD = 0.52;
    const STABLE_FRAMES = 4;            // slightly more stable gesture switching
    const BRUSH_SIZE    = 5;
    const ERASER_SIZE   = 48;
    const DRAG_RADIUS   = 60;

    // Min squared pixel distance between consecutive draw points.
    // Eliminates micro-jitter sticking. ~4px threshold.
    const MIN_DRAW_DIST_SQ = 16;

    // ── Draw color ──────────────────────────────────────────────────────────
    const DRAW_COLOR = "#c084fc"; // violet-400 neon

    // ── Helpers ─────────────────────────────────────────────────────────────

    const dist3d = (a: Lm, b: Lm) =>
      Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

    const handScale = (lms: Lm[]) => {
      const s = Math.max(dist3d(lms[5], lms[17]), dist3d(lms[0], lms[9]));
      return s || 1;
    };

    const fingersUp = (lms: Lm[]): boolean[] => {
      const up: boolean[] = [];
      const palmFacing = lms[5].x < lms[17].x;
      // Thumb: use x-axis
      up.push(palmFacing ? lms[4].x < lms[3].x : lms[4].x > lms[3].x);
      // Fingers: use y-axis (tip above PIP)
      for (let i = 1; i < FINGER_TIPS.length; i++) {
        up.push(lms[FINGER_TIPS[i]].y < lms[FINGER_PIPS[i]].y - 0.01); // small deadzone
      }
      return up;
    };

    const detectGesture = (lms: Lm[], state: PerHandState): GestureName => {
      const up = fingersUp(lms);
      const totalUp = up.filter(Boolean).length;
      const scale = handScale(lms);
      const pinchNorm = dist3d(lms[4], lms[8]) / scale;

      // Hysteretic pinch
      const pinchCandidate =
        up[1] &&
        up.slice(2).filter(Boolean).length <= 1 &&
        ((!state.pinched && pinchNorm < PINCH_ON_THRESHOLD) ||
          (state.pinched && pinchNorm < PINCH_OFF_THRESHOLD));

      state.pinched = !!pinchCandidate;

      let g: GestureName;
      if (state.pinched) {
        g = "PINCH";
      } else if (!up[0] && up[1] && !up[2] && !up[3] && !up[4]) {
        g = "DRAW";
      } else if (!up[0] && up[1] && up[2] && !up[3] && !up[4]) {
        g = "MOVE";
      } else if (totalUp === 5) {
        g = "ERASE";
      } else {
        g = "UNKNOWN";
      }

      if (g === state.lastGesture) {
        state.stableCount += 1;
      } else {
        state.lastGesture = g;
        state.stableCount = 1;
      }

      if (state.stableCount >= STABLE_FRAMES) {
        state.stableGesture = g;
        return g;
      }
      return state.stableGesture;
    };

    // ── Rendering ────────────────────────────────────────────────────────────

    /** Renders a single stroke using Catmull-Rom → converted to Bézier cubic. */
    function renderStroke(s: Stroke) {
      if (s.pts.length < 2 || !ctx) return;

      const drawPath = () => {
        ctx.beginPath();
        ctx.moveTo(s.pts[0].x, s.pts[0].y);

        if (s.pts.length === 2) {
          ctx.lineTo(s.pts[1].x, s.pts[1].y);
        } else {
          // Quadratic Bézier through midpoints for smooth curves
          for (let i = 1; i < s.pts.length - 2; i++) {
            const mx = (s.pts[i].x + s.pts[i + 1].x) / 2;
            const my = (s.pts[i].y + s.pts[i + 1].y) / 2;
            ctx.quadraticCurveTo(s.pts[i].x, s.pts[i].y, mx, my);
          }
          const n = s.pts.length;
          ctx.quadraticCurveTo(
            s.pts[n - 2].x, s.pts[n - 2].y,
            s.pts[n - 1].x, s.pts[n - 1].y
          );
        }
        ctx.stroke();
      };

      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      // Outer glow
      ctx.shadowColor = s.col;
      ctx.shadowBlur = 18;
      ctx.strokeStyle = s.col + "55"; // ~33% opacity glow
      ctx.lineWidth = s.th * 3;
      drawPath();

      // Core line
      ctx.shadowBlur = 8;
      ctx.strokeStyle = s.col;
      ctx.lineWidth = s.th;
      drawPath();

      ctx.shadowBlur = 0;
    }

    function renderCanvas() {
      if (!canvas || !ctx) return;
      const w = canvas.width;
      const h = canvas.height;

      ctx.fillStyle = "#050505";
      ctx.fillRect(0, 0, w, h);

      // All completed strokes
      for (const s of strokes) renderStroke(s);

      for (const hState of handStates) {
        // Live stroke being drawn
        if (hState.currentStroke) renderStroke(hState.currentStroke);

        // Hand skeleton overlay
        if (hState.lastHand2d && hState.lastHand2d.length === 21) {
          ctx.save();
          ctx.strokeStyle = "rgba(200,200,255,0.25)";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          for (const [a, b] of HAND_CONNECTIONS) {
            const pa = hState.lastHand2d[a];
            const pb = hState.lastHand2d[b];
            ctx.moveTo(pa.x, pa.y);
            ctx.lineTo(pb.x, pb.y);
          }
          ctx.stroke();

          // Joint dots
          for (const pt of hState.lastHand2d) {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(180,180,255,0.4)";
            ctx.fill();
          }
          ctx.restore();
        }

        // Cursor ring
        if (hState.smoothedCursor) {
          const cp = hState.smoothedCursor;
          const g = hState.stableGesture;
          const isErasing = g === "ERASE";
          const isMove = g === "MOVE";
          const ringColor = isErasing ? "#f87171" : isMove ? "#94a3b8" : DRAW_COLOR;
          const ringR = isErasing ? ERASER_SIZE : BRUSH_SIZE + 10;

          ctx.save();
          ctx.strokeStyle = ringColor;
          ctx.lineWidth = 2.5;
          ctx.shadowColor = ringColor;
          ctx.shadowBlur = 12;

          // Outer ring
          ctx.beginPath();
          ctx.arc(cp.x, cp.y, ringR, 0, Math.PI * 2);
          ctx.stroke();

          // Inner dot
          ctx.beginPath();
          ctx.arc(cp.x, cp.y, 3, 0, Math.PI * 2);
          ctx.fillStyle = ringColor;
          ctx.fill();

          if (isErasing) {
            ctx.fillStyle = "rgba(248,113,113,0.08)";
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, ERASER_SIZE, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.restore();
        }
      }
    }

    // ── Resize ───────────────────────────────────────────────────────────────
    const resize = () => {
      if (!canvas) return;
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      renderCanvas();
    };

    const onClear = () => {
      strokes = [];
      for (const h of handStates) {
        h.currentStroke = null;
        h.lastPoint = null;
      }
      renderCanvas();
    };

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("clear-canvas", onClear);

    // ── FPS meter ─────────────────────────────────────────────────────────
    let frameCount = 0;
    let lastFpsTime = performance.now();
    const fpsInterval = setInterval(() => {
      const now = performance.now();
      const elapsed = (now - lastFpsTime) / 1000;
      setFps(Math.round(frameCount / elapsed));
      frameCount = 0;
      lastFpsTime = now;
    }, 1000);

    // ── MediaPipe ─────────────────────────────────────────────────────────
    let camera: any = null;
    let hands: any = null;
    let started = false;

    const start = async () => {
      if (started || !canvas || !video) return;
      started = true;

      const [{ Hands }, { Camera }] = await Promise.all([
        import("@mediapipe/hands"),
        import("@mediapipe/camera_utils"),
      ]);

      hands = new Hands({
        locateFile: (file: string) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });

      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,            // keep 1 for real-time speed
        minDetectionConfidence: 0.85,  // ↑ more accurate initial detection
        minTrackingConfidence: 0.80,   // ↑ stabler tracking
      });

      hands.onResults((results: any) => {
        if (!canvas) return;
        frameCount++;
        const w = canvas.width;
        const h = canvas.height;

        const allLmsWorld = results.multiHandLandmarks as Lm[][] | undefined;

        if (!allLmsWorld || allLmsWorld.length === 0) {
          for (const hState of handStates) {
            if (hState.currentStroke && hState.currentStroke.pts.length >= 2)
              strokes.push(hState.currentStroke);
            hState.currentStroke = null;
            hState.smoothedCursor = null;
            hState.lastHand2d = null;
            hState.selectedIdx = null;
            hState.lastPoint = null;
            hState.filter.reset();
          }
          setGesture("UNKNOWN");
          renderCanvas();
          return;
        }

        for (let i = 0; i < handStates.length; i++) {
          const hState = handStates[i];
          const lmsWorld = allLmsWorld[i];

          if (!lmsWorld) {
            if (hState.currentStroke && hState.currentStroke.pts.length >= 2)
              strokes.push(hState.currentStroke);
            hState.currentStroke = null;
            hState.smoothedCursor = null;
            hState.lastHand2d = null;
            hState.selectedIdx = null;
            hState.lastPoint = null;
            hState.filter.reset();
            continue;
          }

          // Map to canvas pixels (mirror x)
          hState.lastHand2d = lmsWorld.map((lm) => ({
            x: (1 - lm.x) * w,
            y: lm.y * h,
          }));

          // ── Tip stabilization ──────────────────────────────────────────
          // Blend index tip (8) with middle pip (7) to dampen rotational jitter.
          // 0.82 / 0.18 is tuned to keep tip precision while smoothing wobble.
          const tip = lmsWorld[8]; // index tip
          const pip = lmsWorld[7]; // index DIP
          const rawX = (1 - (tip.x * 0.82 + pip.x * 0.18)) * w;
          const rawY = (tip.y * 0.82 + pip.y * 0.18) * h;

          hState.smoothedCursor = hState.filter.filter(rawX, rawY, performance.now());

          const g = detectGesture(lmsWorld, hState);
          if (i === 0) setGesture(g);

          const px = hState.smoothedCursor.x;
          const py = hState.smoothedCursor.y;

          if (g === "PINCH") {
            hState.lastPoint = null;
            if (hState.selectedIdx == null) {
              let best: number | null = null;
              let bestD2 = DRAG_RADIUS * DRAG_RADIUS;
              for (let j = 0; j < strokes.length; j++) {
                for (const p of strokes[j].pts) {
                  const d2 = (px - p.x) ** 2 + (py - p.y) ** 2;
                  if (d2 < bestD2) { bestD2 = d2; best = j; }
                }
              }
              hState.selectedIdx = best;
              hState.pinchPrev = { x: px, y: py };
            }
            if (hState.selectedIdx != null && hState.pinchPrev) {
              const dx = px - hState.pinchPrev.x;
              const dy = py - hState.pinchPrev.y;
              if (dx || dy) {
                strokes[hState.selectedIdx].pts =
                  strokes[hState.selectedIdx].pts.map((p) => ({
                    x: p.x + dx,
                    y: p.y + dy,
                  }));
              }
              hState.pinchPrev = { x: px, y: py };
            }
            if (hState.currentStroke && hState.currentStroke.pts.length >= 2)
              strokes.push(hState.currentStroke);
            hState.currentStroke = null;

          } else if (g === "DRAW") {
            hState.selectedIdx = null;

            if (!hState.currentStroke) {
              hState.currentStroke = {
                pts: [{ x: px, y: py }],
                col: DRAW_COLOR,
                th: BRUSH_SIZE,
              };
              hState.lastPoint = { x: px, y: py };
            } else {
              // ── Minimum-distance gating ──────────────────────────────
              // Only append a new point if the cursor moved at least
              // MIN_DRAW_DIST_SQ pixels². This eliminates the "stuck /
              // jittery" artefact where micro-jitter piles up points
              // in the same place and the stroke looks jagged.
              const lp = hState.lastPoint;
              if (!lp) {
                hState.currentStroke.pts.push({ x: px, y: py });
                hState.lastPoint = { x: px, y: py };
              } else {
                const d2 = (px - lp.x) ** 2 + (py - lp.y) ** 2;
                if (d2 >= MIN_DRAW_DIST_SQ) {
                  hState.currentStroke.pts.push({ x: px, y: py });
                  hState.lastPoint = { x: px, y: py };
                }
              }
            }

          } else if (g === "ERASE") {
            hState.selectedIdx = null;
            hState.lastPoint = null;
            if (hState.currentStroke && hState.currentStroke.pts.length >= 2)
              strokes.push(hState.currentStroke);
            hState.currentStroke = null;

            const rr2 = ERASER_SIZE * ERASER_SIZE;
            const newStrokes: Stroke[] = [];
            for (const s of strokes) {
              let batch: { x: number; y: number }[] = [];
              for (const p of s.pts) {
                if ((p.x - px) ** 2 + (p.y - py) ** 2 > rr2) {
                  batch.push(p);
                } else {
                  if (batch.length >= 2) newStrokes.push({ ...s, pts: batch });
                  batch = [];
                }
              }
              if (batch.length >= 2) newStrokes.push({ ...s, pts: batch });
            }
            strokes = newStrokes;

          } else {
            hState.lastPoint = null;
            if (hState.currentStroke && hState.currentStroke.pts.length >= 2)
              strokes.push(hState.currentStroke);
            hState.currentStroke = null;
            hState.selectedIdx = null;
          }
        }

        renderCanvas();
      });

      // ── Camera: higher resolution for better landmark quality ──────────
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 60 },   // request high framerate for smoothness
        },
        audio: false,
      });
      video.srcObject = stream;
      try {
        await video.play();
      } catch (err: any) {
        if (err?.name !== "AbortError") console.error("video.play() failed", err);
      }

      camera = new Camera(video, {
        onFrame: async () => {
          if (hands) await hands.send({ image: video });
        },
        width: 1280,
        height: 720,
      });
      camera.start();
    };

    start().catch((err) => console.error("Error starting MediaPipe", err));

    return () => {
      clearInterval(fpsInterval);
      window.removeEventListener("resize", resize);
      window.removeEventListener("clear-canvas", onClear);
      if (camera && camera.stop) camera.stop();
      if (video.srcObject)
        (video.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
    };
  }, []);

  const gestureColor = GESTURE_COLORS[gesture];

  return (
    <main className="w-screen h-screen bg-black text-white overflow-hidden relative">
      <video ref={videoRef} className="hidden" playsInline autoPlay muted />
      <canvas ref={canvasRef} className="block h-full w-full" />

      {/* ── Top Bar ────────────────────────────────────────────────────── */}
      <div className="pointer-events-none fixed inset-x-0 top-0 flex justify-between items-start px-4 py-3 sm:px-6 sm:py-4 gap-3">
        {/* Left: Title + gesture */}
        <div className="space-y-1.5">
          <div
            style={{ borderColor: gestureColor + "55" }}
            className="rounded-2xl border bg-black/60 px-4 py-2.5 backdrop-blur shadow-lg shadow-black/60"
          >
            <div className="text-sm font-bold tracking-widest text-white/80 uppercase">
              Air Draw
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span
                style={{ backgroundColor: gestureColor }}
                className="h-2 w-2 rounded-full transition-colors duration-300"
              />
              <span
                style={{ color: gestureColor }}
                className="text-xs font-semibold tracking-wider uppercase transition-colors duration-300"
              >
                {gesture}
              </span>
              <span className="ml-3 text-[10px] text-zinc-600 font-mono">
                {fps} fps
              </span>
            </div>
          </div>
        </div>

        {/* Right: Buttons + guide */}
        <div className="flex flex-col items-end gap-3 pointer-events-auto">
          <div className="flex gap-2">
            <button
              onClick={handleClear}
              className="flex items-center gap-2 rounded-full bg-zinc-800/80 px-4 py-2 text-sm font-semibold text-zinc-300 transition-all hover:bg-zinc-700 hover:text-white backdrop-blur"
            >
              <Trash2 className="h-4 w-4" />
              <span>Clear</span>
            </button>
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 rounded-full bg-zinc-100 px-4 py-2 text-sm font-semibold text-zinc-900 transition-all hover:bg-white hover:shadow-[0_0_15px_rgba(255,255,255,0.25)]"
            >
              <Download className="h-4 w-4" />
              <span>Save</span>
            </button>
            <a
              href="https://github.com/NotShubham1112/AIRDRAW"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 rounded-full bg-zinc-800/80 px-4 py-2 text-sm font-semibold text-zinc-300 transition-all hover:bg-zinc-700 hover:text-white backdrop-blur border border-white/10"
            >
              <Github className="h-4 w-4" />
              <span>GitHub</span>
            </a>
          </div>

          <div className="hidden md:flex flex-col items-end rounded-2xl bg-black/60 px-4 py-2.5 backdrop-blur text-[11px] text-zinc-400 space-y-0.5 border border-white/5">
            <span>☝️ DRAW – index only</span>
            <span>✌️ MOVE – peace sign</span>
            <span>🖐 ERASE – open palm</span>
            <span>🤌 PINCH – grab & drag</span>
          </div>
        </div>
      </div>

      {/* ── Bottom hint ───────────────────────────────────────────────── */}
      <div className="pointer-events-none fixed inset-x-0 bottom-3 flex justify-center">
        <span className="rounded-full bg-black/60 px-4 py-1.5 text-[11px] text-zinc-500 backdrop-blur border border-white/5">
          Hold your hand steady · precise tracking active
        </span>
      </div>
    </main>
  );
}