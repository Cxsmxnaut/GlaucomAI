// Alignment is a demonstration aid, not a medical quality or liveness test.
export class HoldTimer {
  constructor(duration = 10000, maxGap = 400) {
    this.duration = duration;
    this.maxGap = maxGap;
    this.reset();
  }
  reset() { this.start = null; this.last = null; }
  update(now, acceptable) {
    if (!Number.isFinite(now) || !acceptable ||
        (this.last !== null && (now < this.last || now - this.last > this.maxGap))) {
      this.reset();
    }
    if (!Number.isFinite(now) || !acceptable) return { progress: 0, remaining: Math.ceil(this.duration / 1000), complete: false };
    if (this.start === null) this.start = now;
    this.last = now;
    const elapsed = Math.max(0, now - this.start);
    return { progress: Math.min(1, elapsed / this.duration),
      remaining: Math.ceil(Math.max(0, this.duration - elapsed) / 1000), complete: elapsed >= this.duration };
  }
}

export function facePosition(points, width, height) {
  if (!Array.isArray(points) || points.length < 468 ||
      !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0 ||
      !Array.from(points).every(p => p && Number.isFinite(p.x) && Number.isFinite(p.y))) {
    return { aligned: false, cue: "Bring your face into view" };
  }
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const box = { x: Math.min(...xs), y: Math.min(...ys), right: Math.max(...xs), bottom: Math.max(...ys) };
  box.width = box.right - box.x;
  box.height = box.bottom - box.y;
  const center = { x: (box.x + box.right) / 2, y: (box.y + box.bottom) / 2 };
  const eye = { x: (points[33].x + points[133].x) / 2, y: (points[33].y + points[133].y) / 2,
    width: Math.abs(points[133].x - points[33].x) };
  const otherEye = { x: (points[263].x + points[362].x) / 2, y: (points[263].y + points[362].y) / 2 };
  const roll = Math.abs(Math.atan2((otherEye.y - eye.y) * height,
    Math.abs(otherEye.x - eye.x) * width));
  let cue = "Hold steady";
  if (box.width < .24) cue = "Move a little closer";
  else if (box.width > .68 || box.x < .02 || box.right > .98 || box.y < .01 || box.bottom > .99) cue = "Move slightly back";
  else if (Math.abs(center.x - .5) > .13 || Math.abs(center.y - .48) > .17) cue = "Center your face in the guide";
  else if (roll > .21 || eye.width < .03) cue = "Keep your eyes level";
  return { aligned: cue === "Hold steady", cue, box, center, eye };
}

export class MotionWindow {
  constructor() { this.samples = []; }
  reset() { this.samples = []; }
  update(now, position) {
    if (!Number.isFinite(now) || !Number.isFinite(position?.center?.x) ||
        !Number.isFinite(position?.center?.y) || !Number.isFinite(position?.box?.width)) {
      this.reset();
      return false;
    }
    if (this.samples.length && now < this.samples.at(-1).time) this.reset();
    this.samples = this.samples.filter(sample => now - sample.time <= 500);
    this.samples.push({ time: now, x: position.center.x, y: position.center.y, size: position.box.width });
    if (this.samples.length < 3 || now - this.samples[0].time < 250) return false;
    const spread = key => Math.max(...this.samples.map(p => p[key])) - Math.min(...this.samples.map(p => p[key]));
    return spread("x") < .018 && spread("y") < .018 && spread("size") < .03;
  }
}
