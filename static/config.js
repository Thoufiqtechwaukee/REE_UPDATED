/* Where the FastAPI backend lives.
 *
 * Leave it empty when this page is served by the backend itself (a local
 * `uvicorn app:app` run) - every call then goes to the same origin.
 *
 * Set it to the full origin of the API when the page is hosted somewhere else,
 * with no trailing slash and no /api suffix:
 *
 *     window.API_BASE = "https://resume-depth.onrender.com";
 *
 * That origin must also be listed in the backend's ALLOWED_ORIGINS, or the
 * browser will block the call before it is ever sent.
 */
window.API_BASE = "";
