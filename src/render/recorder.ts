/** Canvas capture: WebM video recording (MediaRecorder over captureStream)
 * and PNG screenshots, both saved via the browser's download flow. */

function stamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_` +
         `${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

export function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);   // detached-anchor clicks can silently no-op
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function screenshot(canvas: HTMLCanvasElement, prefix = "r1-sim") {
  canvas.toBlob((b) => { if (b) download(b, `${prefix}_${stamp()}.png`); }, "image/png");
}

export class CanvasRecorder {
  private rec: MediaRecorder | null = null;
  private chunks: Blob[] = [];

  get recording(): boolean { return this.rec !== null; }

  start(canvas: HTMLCanvasElement, fps = 60) {
    if (this.rec) return;
    const stream = canvas.captureStream(fps);
    const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]
      .find((m) => MediaRecorder.isTypeSupported(m));
    this.chunks = [];
    this.rec = new MediaRecorder(stream, {
      mimeType: mime, videoBitsPerSecond: 12_000_000,
    });
    this.rec.ondataavailable = (e) => { if (e.data.size) this.chunks.push(e.data); };
    this.rec.start(250);              // gather chunks continuously
  }

  /** Stop and save; resolves once the file download has been triggered. */
  stop(prefix = "r1-sim"): Promise<void> {
    return new Promise((resolve) => {
      const rec = this.rec;
      if (!rec) return resolve();
      this.rec = null;
      rec.onstop = () => {
        download(new Blob(this.chunks, { type: rec.mimeType || "video/webm" }),
                 `${prefix}_${stamp()}.webm`);
        this.chunks = [];
        resolve();
      };
      rec.stop();
    });
  }
}
