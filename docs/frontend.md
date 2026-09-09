# Frontend

The frontend is deliberately build-free: static HTML, CSS, and ES modules.
`api.js` reads `frontend/config.js`, whose empty `apiBase` keeps local FastAPI
development same-origin. The Pages workflow replaces it with the public
Render URL during deployment.

Playback state stays in the controllers. 2D grids arrive as packed bits and
3D voxels as `Uint8Array` coordinate triples; Canvas and Three.js decode those
buffers locally for drawing. The JSON state endpoints carry metadata only;
explicit save/export endpoints remain available when a complete state is
needed.
