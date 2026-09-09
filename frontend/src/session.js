function newSessionSuffix() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

export function getSessionId(kind) {
  const storageKey = `emergent-${kind}-session`;
  try {
    const existing = globalThis.sessionStorage?.getItem(storageKey);
    if (existing) return existing;
  } catch {
    // Private browsing settings can make sessionStorage unavailable.
  }

  const identifier = `${kind}-${newSessionSuffix()}`;
  try {
    globalThis.sessionStorage?.setItem(storageKey, identifier);
  } catch {
    // A generated in-memory identifier is still isolated for this page load.
  }
  return identifier;
}
