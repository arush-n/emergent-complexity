# Deployment

Public frontend URL: <https://arush-n.github.io/emergent-complexity/>

GitHub Pages serves the static frontend; it cannot run JAX or FastAPI. The
included `render.yaml` deploys the backend container to Render, and
`.github/workflows/pages.yml` publishes `frontend/` to Pages.

After the Render service has a stable HTTPS URL, create a repository variable
named `PUBLIC_API_BASE` with that URL. The Pages workflow writes it to
`frontend/config.js` before publishing. Render is configured to allow the
GitHub Pages origin through `EMERGENT_CORS_ORIGINS`.

The local full-stack URL remains `http://127.0.0.1:8000`. Sessions are bounded,
expiring in-memory objects; there is no database or cross-process session
coordination.
