import { Platform } from "react-native";

import type { PickedDocument } from "./filePicker";

export type OcrOptions = {
  forceAi: boolean;
  fontAiAudit: boolean;
  textRemovalMode: string;
};

export type ReconstructOptions = {
  targetLang: string;
  style: string;
  tone: string;
  debugCompare: boolean;
  exportHtml: boolean;
};

type BackendDiscoveryResult = {
  baseUrl: string;
  payload: any;
};

function trimBaseUrl(value: string) {
  return (value || "").trim().replace(/\/+$/, "");
}

function buildUrl(baseUrl: string, path: string, params?: Record<string, string | boolean | undefined>) {
  const url = new URL(`${trimBaseUrl(baseUrl)}${path}`);
  for (const [key, rawValue] of Object.entries(params || {})) {
    if (rawValue === undefined || rawValue === "") {
      continue;
    }
    url.searchParams.set(key, typeof rawValue === "boolean" ? String(rawValue) : rawValue);
  }
  return url.toString();
}

function appendFileToFormData(formData: FormData, file: PickedDocument) {
  if (Platform.OS === "web" && file.webFile) {
    formData.append("file", file.webFile, file.name);
    return;
  }
  formData.append("file", {
    uri: file.uri,
    name: file.name,
    type: file.mimeType || "application/octet-stream",
  } as any);
}

function withTimeout(ms: number) {
  if (typeof AbortController === "undefined") {
    return { signal: undefined as AbortSignal | undefined, cancel: () => {} };
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return {
    signal: controller.signal,
    cancel: () => clearTimeout(timer),
  };
}

function candidateBaseUrls(preferred?: string) {
  const candidates: string[] = [];
  const push = (value?: string) => {
    const trimmed = trimBaseUrl(value || "");
    if (trimmed && !candidates.includes(trimmed)) {
      candidates.push(trimmed);
    }
  };

  push(preferred);

  if (Platform.OS === "web" && typeof window !== "undefined") {
    const hostname = window.location.hostname || "127.0.0.1";
    push(`${window.location.protocol}//${hostname}:8001`);
    push(`http://${hostname}:8001`);
  }

  push("http://127.0.0.1:8001");
  push("http://localhost:8001");

  if (Platform.OS !== "web") {
    push("http://10.0.2.2:8001");
  }

  return candidates;
}

async function ensureJson(response: Response) {
  const text = await response.text();
  const contentType = (response.headers.get("content-type") || "").toLowerCase();
  const looksLikeHtml = contentType.includes("text/html") || /^\s*<!doctype html/i.test(text) || /^\s*<html/i.test(text);
  if (looksLikeHtml) {
    throw new Error(
      "L'URL API pointe vers une page HTML et non vers le backend FastAPI. Verifie que `ocr_server.py` tourne bien sur le bon port."
    );
  }
  let payload: any = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = { raw: text };
  }
  if (!response.ok) {
    throw new Error(payload?.error || `HTTP ${response.status}`);
  }
  return payload;
}

export async function runOcrRequest(baseUrl: string, file: PickedDocument, options: OcrOptions) {
  const url = buildUrl(baseUrl, "/ocr", {
    force_ai: options.forceAi,
    font_ai_audit: options.fontAiAudit,
    text_removal_mode: options.textRemovalMode,
  });
  const formData = new FormData();
  appendFileToFormData(formData, file);
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });
  return ensureJson(response);
}

export async function runReconstructRequest(baseUrl: string, pages: any[], options: ReconstructOptions) {
  const url = buildUrl(baseUrl, "/reconstruct", {
    target_lang: options.targetLang,
    debug_compare: options.debugCompare,
    export_html: options.exportHtml,
    style: options.style,
    tone: options.tone,
  });
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ pages }),
  });
  return ensureJson(response);
}

export async function pingBackend(baseUrl: string) {
  const timeout = withTimeout(2200);
  try {
    const response = await fetch(buildUrl(baseUrl, "/healthz"), {
      method: "GET",
      signal: timeout.signal,
    });
    return await ensureJson(response);
  } finally {
    timeout.cancel();
  }
}

export async function discoverBackendBaseUrl(preferred?: string): Promise<BackendDiscoveryResult> {
  const tried: string[] = [];
  for (const candidate of candidateBaseUrls(preferred)) {
    tried.push(candidate);
    try {
      const payload = await pingBackend(candidate);
      if (payload?.status === "ok") {
        return {
          baseUrl: candidate,
          payload,
        };
      }
    } catch {
      continue;
    }
  }
  throw new Error(`Backend FastAPI introuvable. Candidats testes: ${tried.join(", ")}`);
}
