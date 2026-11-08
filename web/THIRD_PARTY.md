# Browser assets and example image

- Face tracking uses Google's `@mediapipe/tasks-vision` **1.2.2**, listed as Apache-2.0 in its npm package metadata. [Official repository](https://github.com/google-ai-edge/mediapipe), [SDK documentation](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Web).
- The Face Landmarker model is fetched from Google's official versioned model location. Review the publisher's model documentation and terms before redistribution. The tracker maps face/eye geometry; it is not a retinal camera, liveness verifier, or medical model.
- Exact URLs, versions, sizes, SDK archive integrity and individual SHA-256 hashes are pinned in `assets.lock.json`. The setup script verifies every asset. Downloaded vendor assets are ignored by Git and served locally; camera frames are not sent to Google or the Python server.
- The single retinal example comes from Riley Kiefer's EyePACS-AIROGS-light-V2 **version 4**, test/RG sample `EyePACS-TRAIN-RG-3100.jpg`. Its checksum is pinned. It remains in ignored local data storage and is served through a metadata-stripping preview route. [Source terms and attribution](../DATASETS.md).
- The example is never represented as coming from the webcam. Analysis uses the original published EfficientNet-B0 research checkpoint, with its identity exposed by `/api/model`.
