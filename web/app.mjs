import { HoldTimer, MotionWindow, facePosition } from './hold-state.mjs';

const $ = id => document.getElementById(id);
const timer = new HoldTimer(), motion = new MotionWindow();
let FaceLandmarker, tracker, model, stream, frame, zoomTimeout, imageURL;
let session = 0, request, phase = 'idle', lastVideoTime = -1, lastDetection = 0;
const video = $('camera'), overlay = $('landmarks'), capture = $('capture');
const context = overlay.getContext('2d');
const display = (id, visible) => { $(id).hidden = !visible; };
const say = (id, value) => { $(id).textContent = value; };
const fail = message => { say('error', message); display('error', true); };

function step(number) {
  document.querySelectorAll('[data-step]').forEach(item => {
    const current = Number(item.dataset.step);
    item.classList.toggle('done', current < number);
    if (current === number) item.setAttribute('aria-current', 'step');
    else item.removeAttribute('aria-current');
  });
  say('panel-step', `STEP ${String(number).padStart(2, '0')}`);
}

function stopCamera() {
  cancelAnimationFrame(frame);
  stream?.getTracks().forEach(track => track.stop());
  stream = undefined;
  video.pause();
  video.srcObject = null;
  $('stage-dot').classList.remove('active');
}

function buttons() {
  $('camera-button').disabled = !tracker || phase !== 'idle';
  $('example-button').disabled = !model?.ready || !model?.example_available || phase !== 'idle';
  $('upload-button').disabled = !model?.ready;
}

function reset() {
  ++session;
  request?.abort();
  clearTimeout(zoomTimeout);
  stopCamera();
  if (imageURL) URL.revokeObjectURL(imageURL);
  imageURL = undefined;
  $('retinal-image').removeAttribute('src');
  capture.getContext('2d').clearRect(0, 0, capture.width, capture.height);
  context.clearRect(0, 0, overlay.width, overlay.height);
  capture.style.transform = '';
  capture.style.transformOrigin = '';
  timer.reset(); motion.reset(); lastVideoTime = -1; lastDetection = 0;
  phase = 'idle';
  $('camera-stage').classList.remove('steady', 'processing');
  for (const id of ['camera', 'landmarks', 'capture', 'retinal-image', 'alignment-guide',
    'frame-cue', 'example-watermark', 'error', 'results']) display(id, false);
  for (const id of ['camera-placeholder', 'instructions', 'timer', 'alignment-checks',
    'camera-button', 'example-button']) display(id, true);
  for (const id of ['face-check', 'eyes-check', 'steady-check']) $(id).classList.remove('passed');
  say('stage-source', 'Camera preview · off');
  say('privacy-caption', 'Camera frames are processed on this device');
  say('capture-caption', 'Good lighting and a comfortable position make it easier to align.');
  say('instruction-title', "Let's find your comfortable position.");
  say('instruction-copy', "Sit facing your camera, with your whole face in view. We'll guide you from there.");
  say('session-badge', 'Ready when you are');
  say('camera-button-text', 'Enable camera');
  say('seconds', '10');
  $('timer-progress').style.strokeDashoffset = '358.142';
  $('fundus-upload').value = '';
  step(1); buttons();
}

async function jsonResponse(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Analysis is unavailable. No score was produced.');
  return data;
}

async function prepare() {
  try {
    model = await jsonResponse(await fetch('/api/model', { cache: 'no-store' }));
    say('model-version', model.ready ? model.version : 'Checkpoint unavailable');
    $('model-dot').classList.toggle('active', model.ready);
    say('about-version', model.version || 'No compatible checkpoint loaded');
    say('about-checksum', model.sha256 || 'Unavailable');
    const evaluation = model.evaluation;
    say('about-evaluation', evaluation ? `${evaluation.samples} test images · ${(evaluation.sensitivity * 100).toFixed(1)}% sensitivity · ${(evaluation.specificity * 100).toFixed(1)}% specificity. Patient separation was not independently verified.` : 'No verified benchmark for this checkpoint.');
    if (!model.ready) fail('The local model is unavailable. Check the server terminal and model path.');
  } catch {
    fail('The local model server could not be reached. Restart the page after starting web_app.py.');
  }
  buttons();
  try {
    const visionModule = await import('/assets/vendor/vision_bundle.mjs');
    FaceLandmarker = visionModule.FaceLandmarker;
    const { FilesetResolver } = visionModule;
    const vision = await FilesetResolver.forVisionTasks('/assets/vendor/wasm');
    tracker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: '/assets/vendor/face_landmarker.task', delegate: 'CPU' },
      runningMode: 'VIDEO', numFaces: 2, minFaceDetectionConfidence: .6,
      minFacePresenceConfidence: .6, minTrackingConfidence: .6,
      outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false,
    });
    say('tracker-status', 'Local face tracking ready · camera starts only when you choose');
  } catch {
    say('tracker-status', 'Face tracking unavailable. You can still explore the retinal example.');
  }
  buttons();
}

function drawLandmarks(points) {
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  context.clearRect(0, 0, overlay.width, overlay.height);
  context.strokeStyle = '#c7efba'; context.lineWidth = Math.max(1, overlay.width / 650);
  const groups = [FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
    FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
    FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS];
  for (const connections of groups) {
    context.beginPath();
    for (const edge of connections) {
      const start = points[edge.start], end = points[edge.end];
      if (!start || !end) continue;
      context.moveTo(start.x * overlay.width, start.y * overlay.height);
      context.lineTo(end.x * overlay.width, end.y * overlay.height);
    }
    context.stroke();
  }
}

function loop(now) {
  if (phase !== 'camera') return;
  frame = requestAnimationFrame(loop);
  if (!video.videoWidth || video.currentTime === lastVideoTime || now - lastDetection < 80) return;
  lastVideoTime = video.currentTime; lastDetection = now;
  let faces;
  try { faces = tracker.detectForVideo(video, now).faceLandmarks; }
  catch {
    reset(); fail('Face tracking stopped. Start again or explore the retinal example.'); return;
  }
  context.clearRect(0, 0, overlay.width, overlay.height);
  const position = faces.length === 1 ? facePosition(faces[0], video.videoWidth, video.videoHeight) :
    { aligned: false, cue: faces.length > 1 ? 'Only one person should be in view' : 'Bring your face into view' };
  if (faces.length === 1) drawLandmarks(faces[0]);
  const steady = position.aligned && motion.update(now, position);
  if (!position.aligned) motion.reset();
  const hold = timer.update(now, steady);
  $('camera-stage').classList.toggle('steady', steady);
  $('face-check').classList.toggle('passed', position.aligned);
  $('eyes-check').classList.toggle('passed', position.aligned);
  $('steady-check').classList.toggle('passed', steady);
  say('frame-cue-text', position.aligned && !steady ? 'Settle into a comfortable, steady position' : position.cue);
  say('seconds', String(hold.remaining));
  $('timer-progress').style.strokeDashoffset = String(358.142 * (1 - hold.progress));
  step(steady ? 2 : 1);
  say('session-badge', steady ? 'Steady hold in progress' : 'Finding your position');
  say('instruction-title', steady ? 'Just a little stillness.' : "Let's find your comfortable position.");
  say('instruction-copy', steady ? 'Breathe normally and keep looking toward your camera. Movement restarts the ten-second hold.' : 'Keep your face centered and your eyes level. The timer starts when you settle.');
  if (hold.complete) closeUp(position.eye);
}

async function startCamera() {
  if (!tracker || phase !== 'idle') return;
  reset(); phase = 'starting'; buttons();
  const ownSession = session;
  say('camera-button-text', 'Waiting for camera…');
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
    if (ownSession !== session) { acquired.getTracks().forEach(track => track.stop()); return; }
    stream = acquired;
    for (const track of stream.getVideoTracks()) track.addEventListener('ended', () => {
      if (phase === 'camera') { reset(); fail('The camera disconnected. Connect it and start again.'); }
    }, { once: true });
    video.srcObject = stream;
    await video.play();
    if (ownSession !== session) return;
    phase = 'camera';
    display('camera-placeholder', false);
    for (const id of ['camera', 'landmarks', 'alignment-guide', 'frame-cue']) display(id, true);
    display('camera-button', false); display('example-button', false);
    $('stage-dot').classList.add('active');
    say('stage-source', 'Live webcam · external eye');
    say('tracker-status', 'Face and eye landmarks are processed locally');
    frame = requestAnimationFrame(loop);
  } catch (error) {
    if (ownSession !== session) return;
    reset();
    const messages = {
      NotAllowedError: 'Camera access was not allowed. Enable it in your browser, or explore the retinal example.',
      NotFoundError: 'No webcam was found. Connect a camera, or explore the retinal example.',
      NotReadableError: 'Your camera is busy or unavailable. Close other camera apps and try again.',
    };
    fail(messages[error.name] || 'The camera could not start. You can still explore the retinal example.');
  }
}

function closeUp(eye) {
  phase = 'closeup';
  capture.width = video.videoWidth; capture.height = video.videoHeight;
  const snapshot = capture.getContext('2d');
  snapshot.translate(capture.width, 0);
  snapshot.scale(-1, 1);
  snapshot.drawImage(video, 0, 0);
  stopCamera();
  for (const id of ['camera', 'landmarks', 'alignment-guide', 'frame-cue', 'timer', 'alignment-checks']) display(id, false);
  display('capture', true);
  $('camera-stage').classList.remove('steady');
  step(3);
  say('stage-source', 'External eye close-up · camera stopped');
  say('privacy-caption', 'This frame stays in your browser and is not analyzed');
  say('capture-caption', 'This is the outside of your eye. Next, a separately supplied example retina.');
  say('instruction-title', 'A closer look at your eye.');
  say('instruction-copy', 'This close-up shows the visible eye. A webcam cannot photograph the retina. We will switch to a separate example retinal image.');
  say('session-badge', 'Webcam demonstration');
  const box = $('camera-stage').getBoundingClientRect();
  const scale = Math.max(box.width / capture.width, box.height / capture.height);
  const x = (box.width - capture.width * scale) / 2 + (1 - eye.x) * capture.width * scale;
  const y = (box.height - capture.height * scale) / 2 + eye.y * capture.height * scale;
  capture.style.transformOrigin = `${100 * x / box.width}% ${100 * y / box.height}%`;
  const ownSession = session;
  zoomTimeout = setTimeout(() => {
    if (ownSession !== session) return;
    capture.style.transform = 'scale(2.7)';
    zoomTimeout = setTimeout(() => { if (ownSession === session) showExample(); }, 2600);
  }, 60);
}

function analysisView(example) {
  phase = 'analysis'; buttons(); step(4);
  for (const id of ['camera-placeholder', 'camera', 'landmarks', 'capture', 'timer', 'alignment-checks', 'camera-button', 'example-button']) display(id, false);
  display('instructions', true); display('results', false); display('example-watermark', example);
  $('camera-stage').classList.add('processing');
  say('stage-source', example ? 'Benchmark example · separate retinal photo' : 'Uploaded retinal photograph');
  say('privacy-caption', 'Retinal image analyzed in memory on your local server');
  say('capture-caption', example ? 'A real EyePACS benchmark photograph. This is not your retina and did not come from your webcam.' : 'Uploaded research image. File checks do not establish retinal image suitability.');
  say('instruction-title', 'Looking at the retinal photo.');
  say('instruction-copy', example ? 'The local model is analyzing a separately supplied benchmark image. Your webcam frame is never sent to this model.' : 'The local model is analyzing your uploaded photograph. Images are not saved by the app.');
  say('session-badge', example ? 'Labeled example image' : 'Research image upload');
}

function showResult(data, expectedKind) {
  if (data.source?.kind !== expectedKind || !Number.isFinite(data.referable) ||
      !Number.isFinite(data.non_referable) || !Number.isFinite(data.threshold) ||
      data.referable < 0 || data.referable > 1) throw new Error('The server returned an invalid result. No score is available.');
  phase = 'result'; step(5);
  $('camera-stage').classList.remove('processing');
  display('instructions', false); display('results', true);
  say('result-source', data.source.label);
  say('score', (data.referable * 100).toFixed(1));
  say('negative-score', `${(data.non_referable * 100).toFixed(1)}%`);
  say('threshold', `${(data.threshold * 100).toFixed(1)}%`);
  $('score-fill').style.width = `${data.referable * 100}%`;
  $('threshold-marker').style.left = `${data.threshold * 100}%`;
  say('decision', data.flagged ? 'Above the research referral threshold. This does not establish glaucoma.' : 'Below the research referral threshold. This does not rule out glaucoma.');
  say('inference-time', `${Math.round(data.elapsed_ms)} ms`);
  say('image-size', `${data.width} × ${data.height}`);
}

async function analyzeFlow(example, file) {
  reset();
  const ownSession = session;
  request = new AbortController();
  const signal = request.signal;
  analysisView(example);
  try {
    let image;
    if (example) {
      const response = await fetch('/api/example', { signal });
      if (!response.ok) await jsonResponse(response);
      image = await response.blob();
    } else image = file;
    if (ownSession !== session) return;
    imageURL = URL.createObjectURL(image);
    $('retinal-image').src = imageURL;
    await $('retinal-image').decode();
    if (ownSession !== session) return;
    display('retinal-image', true);
    const url = example ? '/api/analyze-example' : `/api/analyze?source=fundus_upload&filename=${encodeURIComponent(file.name)}`;
    const response = await fetch(url, { method: 'POST', signal,
      ...(example ? {} : { body: file, headers: { 'Content-Type': 'application/octet-stream' } }) });
    const data = await jsonResponse(response);
    if (ownSession !== session) return;
    showResult(data, example ? 'benchmark_sample' : 'fundus_upload');
  } catch (error) {
    if (ownSession !== session || error.name === 'AbortError') return;
    $('camera-stage').classList.remove('processing');
    say('instruction-title', 'No score is available.');
    say('instruction-copy', 'Start again to retry, or choose another retinal photograph.');
    say('session-badge', 'Analysis did not complete');
    fail(error.message || 'Analysis could not complete. No score was produced.');
  }
}

function showExample() { return analyzeFlow(true); }
$('camera-button').addEventListener('click', startCamera);
$('example-button').addEventListener('click', showExample);
$('reset-button').addEventListener('click', reset);
$('another-button').addEventListener('click', reset);
$('upload-button').addEventListener('click', () => $('fundus-upload').click());
$('fundus-upload').addEventListener('change', event => {
  const file = event.target.files?.[0];
  if (!file) return;
  if (!/\.(png|jpe?g)$/i.test(file.name) || file.size < 1 || file.size >= 10 * 1024 * 1024) {
    reset(); fail('Choose a non-empty PNG or JPEG retinal photograph smaller than 10 MB.'); return;
  }
  analyzeFlow(false, file);
});
for (const id of ['about-button', 'about-footer']) $(id).addEventListener('click', () => $('about-dialog').showModal());
$('close-about').addEventListener('click', () => $('about-dialog').close());
document.addEventListener('visibilitychange', () => {
  if (document.hidden && ['camera', 'starting', 'closeup'].includes(phase)) {
    reset(); fail('The camera stopped when this tab was hidden. Start again when you are ready.');
  }
});
window.addEventListener('pagehide', reset);
reset();
prepare();
