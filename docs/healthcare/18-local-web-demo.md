# Local web demonstration

Date: November 7, 2026. Status: research demonstration. This UI does not add clinical validation or improve model accuracy.

The user explicitly selected a webcam demonstration with a separate example retinal photograph. Ordinary webcams show the external eye; they cannot obtain a fundus photograph without suitable retinal-imaging optics. The transition must never suggest that a user's webcam frame became their retinal photograph.

## User flow

1. Enable the camera through a browser permission prompt. Audio is never requested.
2. Local MediaPipe Face Landmarker draws face, eye, and iris landmarks. Centering, distance, eye level, and recent positional movement determine the alignment cue. Multiple faces reset the hold.
3. Hold a suitable position for ten continuous seconds. Movement, lost alignment, backwards timestamps, or measurement gaps over 400 ms reset the timer. Hiding the tab clears the camera session.
4. A browser-only snapshot zooms into the visible external eye. The camera stops. This image is never sent to the glaucoma model.
5. The UI switches to a separate EyePACS benchmark image. A persistent label says **EXAMPLE RETINA · NOT FROM YOUR WEBCAM**. The server checks the pinned source hash before serving or analyzing it.
6. Real inference displays the referable/non-referable model scores, validation threshold, inference duration, source, and image dimensions. The score is not a diagnosis or calibrated disease probability.

The example button skips the webcam demonstration. Manual upload is a separate route for retinal research images. File format and uniform-image checks do not establish retinal suitability; there is no trained fundus-content detector or acquisition-quality model here. A caller declaring `fundus_upload` is asserting provenance, not proving it. Neither alignment nor the ten-second hold is a liveness test or medical capture-quality check.

## Active model and image provenance

Default: original published EfficientNet-B0 v0.1.0, SHA-256 `fcbcc55c7349cf602757972dbf69fadf2cd76dd58b169c3ab0731aea8fc11ce6`. Benchmark details appear only when the loaded checkpoint hash matches that record. Other compatible, trusted checkpoints are labeled user-supplied. Mixed-label sequential checkpoints are rejected. The incompatible separate-task architecture cannot be loaded by this entry point. Private Drive checkpoints remain separate.

The example is selected deterministically by filename from the publisher's test/referable partition, not selected for its score. Original source: `test/RG/EyePACS-TRAIN-RG-3100.jpg`, Kaggle `deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/versions/4`. SHA-256 `000f552bce6ac5995b4f068cf0112359f295c51821e81c3171ee15882fdd460b`. The image is a demonstration benchmark, not a new accuracy measurement or clinical finding. See [asset sources and terms](../../web/THIRD_PARTY.md) and [the machine-readable lock](../../web/assets.lock.json).

## Running and data handling

Follow the [README setup](../../README.md#local-webcam-demonstration). The server binds only to `127.0.0.1`. No tunnel, cloud deployment, database, or account is needed. Setup downloads pinned local assets; the page then uses same-origin resources. Neither webcam frames nor landmarks are sent to Google, Kaggle, or an external inference API.

Uploads use a bounded raw request body, not multipart temporary files. The server rejects files at or above 10 MB, decodes PNG/JPEG with the shared pixel limit, removes image metadata, and applies the existing exact-uniform guard. Uploaded images are not written to disk by the app. The benchmark itself is an ignored local setup asset. The server rejects foreign browser origins and non-local hostnames, disables response caching, and serializes inference. It does not make a secure multi-user hospital service; use de-identified research images only.

Reset aborts the current browser request, clears image references and canvas pixels, releases object URLs, stops camera tracks, and hides old scores. A session identifier prevents a late response from replacing a newer session. An inference already running on the server may finish after the browser aborts; no image is saved by that route. OS/browser memory behavior is outside these application-level guarantees.

## Verification and practical limits

Automated checks cover actual CPU inference with synthetic test weights, checkpoint compatibility/readiness, source declarations, sample integrity, bounded uploads, image decoding, uniform-image abstention, metadata stripping, local origin/host checks, and concurrent requests. Node tests cover the ten-second timer, movement/tracking loss, frame gaps, malformed inputs, face alignment, eye location, and motion warmup/reset. Test fixtures do not establish medical accuracy.

Browser checks use the isolated preview, without controlling the user's desktop. The local tracker initializes successfully. Example inference, manual benchmark upload, and uniform-image rejection are checked against real HTTP responses. Camera capture and a complete ten-second hold with a real person's face remain a manual hardware check; they have not been represented as verified by unit tests. No real face image is captured during browser QA.

No Graphify graph or coordinated index is configured in this repository. No model training or clinical experiment was performed for this UI change. Existing notebooks, model weights, thresholds, and evaluation evidence remain the authority for model behavior.
