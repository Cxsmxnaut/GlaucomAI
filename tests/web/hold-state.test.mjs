import assert from 'node:assert/strict';
import test from 'node:test';
import { HoldTimer, MotionWindow, facePosition } from '../../web/hold-state.mjs';

function advance(timer, first, last, interval = 100) {
  let result;
  for (let now = first; now <= last; now += interval) result = timer.update(now, true);
  return result;
}

function landmarks() {
  const points = Array.from({ length: 478 }, () => ({ x: .5, y: .48 }));
  points[0] = { x: .3, y: .18 };
  points[1] = { x: .7, y: .78 };
  points[33] = { x: .39, y: .4 };
  points[133] = { x: .45, y: .4 };
  points[263] = { x: .55, y: .4 };
  points[362] = { x: .61, y: .4 };
  return points;
}

const position = (x = .5, y = .48, width = .4) => ({ center: { x, y }, box: { width } });

test('a continuous ten-second hold completes only at ten seconds', () => {
  const timer = new HoldTimer();
  const almost = advance(timer, 0, 9900);
  assert.equal(almost.complete, false);
  assert.equal(almost.remaining, 1);
  assert.equal(almost.progress, .99);
  assert.deepEqual(timer.update(10000, true), { progress: 1, remaining: 0, complete: true });
  assert.deepEqual(timer.update(10100, true), { progress: 1, remaining: 0, complete: true });
});

test('movement or loss of alignment requires a fresh full hold', () => {
  const timer = new HoldTimer();
  advance(timer, 0, 5900);
  assert.deepEqual(timer.update(6000, false), { progress: 0, remaining: 10, complete: false });
  assert.equal(advance(timer, 6100, 16000).complete, false);
  assert.equal(timer.update(16100, true).complete, true);
});

test('a stalled video cannot contribute time toward the hold', () => {
  const timer = new HoldTimer();
  advance(timer, 0, 5000);
  assert.equal(timer.update(5400, true).progress, .54);
  assert.deepEqual(timer.update(5801, true), { progress: 0, remaining: 10, complete: false });
  assert.equal(advance(timer, 5901, 15701).complete, false);
  assert.equal(timer.update(15801, true).complete, true);
});

test('backwards and invalid clocks discard accumulated hold time', () => {
  const timer = new HoldTimer();
  advance(timer, 0, 900);
  assert.equal(timer.update(800, true).progress, 0);
  assert.equal(timer.update(NaN, true).progress, 0);
  assert.equal(timer.update(1200, true).progress, 0);
  assert.equal(timer.update(Infinity, true).complete, false);
});

test('an explicit reset discards a completed session', () => {
  const timer = new HoldTimer(1000);
  assert.equal(advance(timer, 0, 1000).complete, true);
  timer.reset();
  assert.deepEqual(timer.update(1500, true), { progress: 0, remaining: 1, complete: false });
  assert.equal(timer.update(1600, false).remaining, 1);
});

test('a centered level face returns the eye position used for the close-up', () => {
  const result = facePosition(landmarks(), 1280, 720);
  assert.equal(result.aligned, true);
  assert.equal(result.cue, 'Hold steady');
  assert.equal(result.center.x, .5);
  assert.ok(Math.abs(result.eye.x - .42) < 1e-12);
  assert.equal(result.eye.y, .4);
  assert.ok(Math.abs(result.eye.width - .06) < 1e-12);
});

test('distance, position, and eye tilt produce corrective cues', () => {
  const small = landmarks().map(p => ({ x: .5 + (p.x - .5) * .5, y: p.y }));
  assert.match(facePosition(small, 1280, 720).cue, /closer/);
  const large = landmarks().map(p => ({ x: .5 + (p.x - .5) * 2, y: p.y }));
  assert.match(facePosition(large, 1280, 720).cue, /back/);
  const shifted = landmarks().map(p => ({ x: p.x + .14, y: p.y }));
  assert.match(facePosition(shifted, 1280, 720).cue, /Center/);
  const tilted = landmarks();
  tilted[263].y += .1; tilted[362].y += .1;
  assert.match(facePosition(tilted, 1280, 720).cue, /level/);
});

test('incomplete or invalid landmarks and video dimensions never count as aligned', () => {
  const sparse = new Array(478);
  const nullPoint = landmarks(); nullPoint[33] = null;
  const badPoint = landmarks(); badPoint[0].x = NaN;
  for (const points of [null, [], landmarks().slice(0, 467), sparse, nullPoint, badPoint]) {
    assert.equal(facePosition(points, 1280, 720).aligned, false);
  }
  for (const [width, height] of [[0, 720], [1280, 0], [NaN, 720], [1280, Infinity]]) {
    assert.equal(facePosition(landmarks(), width, height).aligned, false);
  }
});

test('motion needs several measurements over enough time before a hold can start', () => {
  const motion = new MotionWindow();
  assert.equal(motion.update(0, position()), false);
  assert.equal(motion.update(100, position(.501)), false);
  assert.equal(motion.update(200, position(.499)), false);
  assert.equal(motion.update(300, position(.502)), true);
});

test('position or scale movement interrupts steadiness until it leaves the window', () => {
  for (const changed of [position(.53), position(.5, .52), position(.5, .48, .44)]) {
    const motion = new MotionWindow();
    for (const now of [0, 100, 200, 300]) motion.update(now, position());
    assert.equal(motion.update(400, changed), false);
    for (const now of [500, 600, 700, 800, 900]) assert.equal(motion.update(now, position()), false);
    assert.equal(motion.update(1000, position()), true);
  }
});

test('invalid or backwards motion timestamps cannot reuse previous samples', () => {
  const motion = new MotionWindow();
  for (const now of [0, 100, 200, 300]) motion.update(now, position());
  assert.equal(motion.update(100, position()), false);
  assert.equal(motion.update(400, { center: { x: NaN, y: .48 }, box: { width: .4 } }), false);
  assert.equal(motion.update(500, position()), false);
  assert.equal(motion.update(NaN, position()), false);
  assert.equal(motion.update(900, position()), false);
  motion.reset();
  assert.equal(motion.update(1000, position()), false);
});
