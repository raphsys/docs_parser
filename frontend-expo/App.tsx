import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Linking,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
  useWindowDimensions,
} from "react-native";
import { StatusBar } from "expo-status-bar";

import { pickDocument, type PickedDocument } from "./lib/filePicker";
import {
  discoverBackendBaseUrl,
  runOcrRequest,
  runReconstructRequest,
  runTranslateStructureRequest,
  type OcrOptions,
  type ReconstructOptions,
} from "./lib/api";

type PreviewKind = "pdf" | "html" | "image";
type ConnectionState = "detecting" | "connected" | "error";
type TargetLang = "fr" | "en" | "de" | "es";
type ToneOption = "neutre" | "didactique" | "analytique" | "formel";
type StyleOption = "professionnel" | "technique" | "scientifique";
type RemovalMode = "default" | "telea" | "ns";
type ExtractionLevel = "block" | "line" | "phrase" | "span";
type AppPage = "translate" | "inspect";

type InspectorItem = {
  id: string;
  level: ExtractionLevel;
  text: string;
  translatedText?: string;
  role: string;
  alignment: string;
  bbox: number[];
  raw: any;
};

type RelationOverlay = {
  key: string;
  kind: "previous" | "next";
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
};

const API_FALLBACK: string = Platform.select({
  web: "http://127.0.0.1:8001",
  default: "http://10.0.2.2:8001",
} as any) as string;

const TARGET_LANGS: TargetLang[] = ["fr", "en", "de", "es"];
const STYLES: StyleOption[] = ["professionnel", "technique", "scientifique"];
const TONES: ToneOption[] = ["neutre", "didactique", "analytique", "formel"];
const REMOVAL_MODES: RemovalMode[] = ["default", "telea", "ns"];
const EXTRACTION_LEVELS: ExtractionLevel[] = ["block", "line", "phrase", "span"];
const APP_PAGES: AppPage[] = ["translate", "inspect"];

function StatusBadge({
  label,
  tone,
}: {
  label: string;
  tone: "neutral" | "success" | "warning";
}) {
  return (
    <View
      style={[
        styles.statusBadge,
        tone === "success" ? styles.statusBadgeSuccess : null,
        tone === "warning" ? styles.statusBadgeWarning : null,
      ]}
    >
      <Text style={styles.statusBadgeText}>{label}</Text>
    </View>
  );
}

function ChipRow<T extends string>({
  title,
  value,
  options,
  onChange,
}: {
  title: string;
  value: T;
  options: readonly T[];
  onChange: (next: T) => void;
}) {
  return (
    <View style={styles.optionGroup}>
      <Text style={styles.optionLabel}>{title}</Text>
      <View style={styles.chipRow}>
        {options.map((option) => {
          const active = option === value;
          return (
            <Pressable key={option} onPress={() => onChange(option)} style={[styles.chip, active && styles.chipActive]}>
              <Text style={[styles.chipText, active && styles.chipTextActive]}>{option}</Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

function ToggleRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <View style={styles.toggleRow}>
      <Text style={styles.toggleLabel}>{label}</Text>
      <Switch value={value} onValueChange={onChange} trackColor={{ false: "#bfceda", true: "#6fb39c" }} />
    </View>
  );
}

function openDownload(url: string, suggestedName: string) {
  if (Platform.OS === "web" && typeof document !== "undefined") {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = suggestedName;
    anchor.target = "_blank";
    anchor.rel = "noreferrer";
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    return;
  }
  Linking.openURL(url);
}

function PreviewSurface({
  title,
  src,
  kind,
  fallback,
}: {
  title: string;
  src: string | null;
  kind: PreviewKind;
  fallback: string;
}) {
  if (!src) {
    return (
      <View style={styles.previewEmpty}>
        <Text style={styles.previewEmptyTitle}>{fallback}</Text>
        <Text style={styles.previewEmptyBody}>Charge un document et lance le pipeline pour remplir cet espace.</Text>
      </View>
    );
  }

  if (Platform.OS === "web") {
    if (kind === "image") {
      return <img src={src} alt={title} style={{ width: "100%", height: "100%", objectFit: "contain", background: "#fff" } as any} />;
    }
    return <iframe src={src} title={title} style={{ width: "100%", height: "100%", border: "0", background: "#fff" } as any} />;
  }

  return (
    <View style={styles.previewEmpty}>
      <Text style={styles.previewEmptyTitle}>Preview native reduite</Text>
      <Text style={styles.previewEmptyBody}>Ouvre le document exporte pour voir le rendu complet.</Text>
    </View>
  );
}

function PreviewPane({
  title,
  subtitle,
  src,
  kind,
  fallback,
}: {
  title: string;
  subtitle: string;
  src: string | null;
  kind: PreviewKind;
  fallback: string;
}) {
  return (
    <View style={styles.previewCard}>
      <View style={styles.previewHeader}>
        <View style={styles.previewHeaderCopy}>
          <Text style={styles.previewTitle}>{title}</Text>
          <Text style={styles.previewSubtitle}>{subtitle}</Text>
        </View>
        <StatusBadge label={kind.toUpperCase()} tone="neutral" />
      </View>
      <View style={styles.previewViewport}>
        <PreviewSurface title={title} src={src} kind={kind} fallback={fallback} />
      </View>
    </View>
  );
}

function buildAssetUrl(baseUrl: string, rawUrl?: string | null) {
  const value = (rawUrl || "").trim();
  if (!value) {
    return null;
  }
  if (/^https?:\/\//i.test(value)) {
    return value;
  }
  return `${baseUrl.replace(/\/+$/, "")}${value.startsWith("/") ? value : `/${value}`}`;
}

function formatScalar(value: any): string {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/\.?0+$/, "");
  }
  if (typeof value === "boolean") {
    return value ? "oui" : "non";
  }
  if (Array.isArray(value)) {
    return value.map((item) => formatScalar(item)).join(" · ");
  }
  return String(value);
}

function flagSummary(flags: Record<string, any> | undefined) {
  if (!flags || typeof flags !== "object") {
    return "n/a";
  }
  const active = Object.entries(flags)
    .filter(([, enabled]) => Boolean(enabled))
    .map(([name]) => name);
  return active.length ? active.join(", ") : "aucun";
}

function topScoreLabel(scores: Record<string, any> | undefined) {
  if (!scores || typeof scores !== "object") {
    return "n/a";
  }
  const entries = Object.entries(scores)
    .map(([name, value]) => [name, Number(value)] as const)
    .filter(([, value]) => Number.isFinite(value))
    .sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    return "n/a";
  }
  const [name, value] = entries[0];
  return `${name} (${formatScalar(value)})`;
}

function summarizeContinuitySide(side: any) {
  if (!side || typeof side !== "object" || !side.exists) {
    return "aucune";
  }
  const main = side.logical_relation || side.visual_relation || "relation";
  const cont = side.continuation ? "continue" : "rupture";
  const conf = side.confidence !== undefined ? `, conf ${formatScalar(side.confidence)}` : "";
  const resolved = side.resolved_by ? `, via ${side.resolved_by}` : "";
  return `${cont}, ${main}${conf}${resolved}`;
}

function summarizePairRelations(relations: any[] | undefined) {
  if (!Array.isArray(relations) || !relations.length) {
    return "n/a";
  }
  const continuationCount = relations.filter((item) => item?.continuation).length;
  const breakCount = relations.length - continuationCount;
  return `${relations.length} liens, ${continuationCount} continuites, ${breakCount} ruptures`;
}

function getRuleset(raw: any) {
  return raw?.translation_ruleset || raw?.element_ruleset || null;
}

function getRulesetSummary(raw: any) {
  return raw?.translation_rulesets?.summary || raw?.element_rulesets?.summary || null;
}

function topCombinedMode(combinedModes: any[] | undefined) {
  if (!Array.isArray(combinedModes) || !combinedModes.length) {
    return "n/a";
  }
  const first = [...combinedModes]
    .filter((item) => item && typeof item === "object")
    .sort((a, b) => Number(b.score || 0) - Number(a.score || 0))[0];
  if (!first) {
    return "n/a";
  }
  return `${first.mode || "mode"} (${formatScalar(first.score)})`;
}

function AttributeRow({ label, value }: { label: string; value: any }) {
  return (
    <View style={styles.attributeRow}>
      <Text style={styles.attributeLabel}>{label}</Text>
      <Text style={styles.attributeValue}>{formatScalar(value)}</Text>
    </View>
  );
}

function AttributeSection({
  title,
  rows,
}: {
  title: string;
  rows: Array<{ label: string; value: any }>;
}) {
  const visibleRows = rows.filter((row) => row.value !== undefined);
  if (!visibleRows.length) {
    return null;
  }
  return (
    <View style={styles.attributeSection}>
      <Text style={styles.detailKey}>{title}</Text>
      {visibleRows.map((row) => (
        <AttributeRow key={`${title}-${row.label}`} label={row.label} value={row.value} />
      ))}
    </View>
  );
}

function normalizeTextFromItem(item: any) {
  if (!item || typeof item !== "object") {
    return "";
  }
  return (
    item.text ||
    item.line_text ||
    item.phrase_text ||
    item.texte ||
    item.raw_text ||
    ""
  );
}

function getStructureFromPageResult(pageResult: any) {
  if (!pageResult || typeof pageResult !== "object") {
    return null;
  }
  return pageResult.structure && typeof pageResult.structure === "object" ? pageResult.structure : pageResult;
}

function getPageListFromStructurePayload(payload: any) {
  if (!payload || typeof payload !== "object") {
    return [];
  }
  if (Array.isArray(payload?.pages)) {
    return payload.pages;
  }
  if (Array.isArray(payload?.structure?.pages)) {
    return payload.structure.pages;
  }
  if (payload?.structure && typeof payload.structure === "object") {
    return [payload.structure];
  }
  return [];
}

function blockKey(block: any, blockIndex: number) {
  return block?.unit_id || block?.id || `block-${blockIndex}`;
}

function normalizeLineIndices(value: any) {
  if (Array.isArray(value)) {
    return value
      .map((item) => Number(item))
      .filter((item) => Number.isFinite(item))
      .sort((a, b) => a - b);
  }
  if (value === null || value === undefined || value === "") {
    return [];
  }
  const single = Number(value);
  return Number.isFinite(single) ? [single] : [];
}

function formatLineRefs(value: any) {
  const indices = normalizeLineIndices(value);
  if (!indices.length) {
    return "n/a";
  }
  return indices.map((index) => `L${index + 1}`).join(", ");
}

function appendSentenceFragment(base: string, fragment: string) {
  const left = (base || "").trimEnd();
  const right = (fragment || "").trim();
  if (!left) {
    return right;
  }
  if (!right) {
    return left;
  }
  if (/-$/.test(left)) {
    return `${left.slice(0, -1)}${right}`;
  }
  if (/[("'“‘-]$/.test(left) || /^[,.;:!?%)\]}"'”’]/.test(right)) {
    return `${left}${right}`;
  }
  return `${left} ${right}`;
}

function endsSentenceText(text: string) {
  const value = (text || "").trim();
  if (!value) {
    return false;
  }
  const lastToken = value.split(/\s+/).pop() || "";
  if (/^(mr|mrs|ms|dr|prof|etc|e\.g|i\.e|cf|vs|fig|eq|no|vol|al)\.$/i.test(lastToken)) {
    return false;
  }
  return /[.!?…]+["'”’)\]}]*$/.test(value);
}

function shouldBreakOnHardBoundary(currentText: string, line: any, previousLine: any) {
  if (!currentText.trim()) {
    return false;
  }
  if (endsSentenceText(currentText)) {
    return true;
  }
  if (line?.leading_marker || line?.paragraph_break_before) {
    return true;
  }
  const currentIndent = Number(line?.indent_px ?? line?.layout_attributes?.indent_px ?? 0);
  const previousIndent = Number(previousLine?.indent_px ?? previousLine?.layout_attributes?.indent_px ?? 0);
  if (Number.isFinite(currentIndent) && Number.isFinite(previousIndent) && currentIndent - previousIndent > 24) {
    return true;
  }
  return false;
}

function flattenPhraseAttributes(raw: any, fallbackRole: string, fallbackAlignment: string) {
  const parts = [
    ["role", raw?.role || fallbackRole || "unknown"],
    ["alignement", raw?.alignment || raw?.layout_attributes?.horizontal_alignment || fallbackAlignment || "unknown"],
    ["multi_ligne", raw?.multi_line],
    ["lignes", formatLineRefs(raw?.line_indices)],
    ["fragments", raw?.fragment_count],
    ["source", raw?.source_kind || raw?.source],
    ["police", raw?.style_attributes?.font_family_primary],
    ["taille_pt", raw?.style_attributes?.font_size_pt_median],
    ["couleur", raw?.style_attributes?.color_primary],
    ["gras", raw?.style_attributes?.flags_ratio?.bold],
    ["italique", raw?.style_attributes?.flags_ratio?.italic],
    ["majuscules", raw?.style_attributes?.flags_ratio?.uppercase],
    ["monospace", raw?.style_attributes?.flags_ratio?.monospace],
    ["strategie", raw?.translation_strategy],
    ["translatable", raw?.translatable],
  ];
  return parts
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => `${key}:${formatScalar(value)}`)
    .join(" | ");
}

function splitLineTextIntoSentenceChunks(text: string) {
  const value = (text || "").replace(/\s+/g, " ").trim();
  if (!value) {
    return [];
  }
  const chunks: Array<{ text: string; start: number; end: number; endsSentence: boolean }> = [];
  let start = 0;
  const pattern = /[.!?…]+["'”’)\]}]*/g;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(value))) {
    const end = match.index + match[0].length;
    const chunkText = value.slice(start, end).trim();
    const nextPart = value.slice(end);
    const lastToken = chunkText.split(/\s+/).pop() || "";
    const isAbbreviation = /^(mr|mrs|ms|dr|prof|etc|e\.g|i\.e|cf|vs|fig|eq|no|vol|al)\.$/i.test(lastToken);
    if (!chunkText || isAbbreviation) {
      continue;
    }
    if (nextPart && !/^\s+[A-Z"“‘(\[]/.test(nextPart) && nextPart.trim()) {
      continue;
    }
    chunks.push({ text: chunkText, start, end, endsSentence: true });
    start = end;
    while (start < value.length && /\s/.test(value[start])) {
      start += 1;
    }
  }
  if (start < value.length) {
    const tail = value.slice(start).trim();
    if (tail) {
      const tailStart = value.indexOf(tail, start);
      chunks.push({ text: tail, start: Math.max(start, tailStart), end: Math.max(start, tailStart) + tail.length, endsSentence: false });
    }
  }
  return chunks;
}

function approximateFragmentBBox(lineBBox: number[], fullText: string, start: number, end: number) {
  if (!Array.isArray(lineBBox) || lineBBox.length !== 4) {
    return [0, 0, 0, 0];
  }
  const [x0, y0, x1, y1] = lineBBox.map((value) => Number(value || 0));
  const content = (fullText || "").replace(/\s+/g, " ").trim();
  const total = Math.max(1, content.length);
  const safeStart = Math.max(0, Math.min(start, total - 1));
  const safeEnd = Math.max(safeStart + 1, Math.min(end, total));
  const left = x0 + (x1 - x0) * (safeStart / total);
  const right = x0 + (x1 - x0) * (safeEnd / total);
  return [Math.round(left), Math.round(y0), Math.round(Math.max(left + 4, right)), Math.round(y1)];
}

function buildSemanticPhrasesFromBlock(block: any) {
  const lines = [...(block?.lines || [])].sort((a: any, b: any) => {
    const ai = Number(a?.line_index ?? 0);
    const bi = Number(b?.line_index ?? 0);
    return ai - bi;
  });
  const phrases: any[] = [];
  let sentenceIndex = 0;
  let currentFragments: any[] = [];
  let currentText = "";

  const flush = (endReason: string) => {
    const fragments = currentFragments.filter((fragment) => String(fragment?.text || "").trim());
    currentFragments = [];
    const mergedText = currentText.trim();
    currentText = "";
    if (!fragments.length) {
      return;
    }
    const text = mergedText || fragments.map((fragment) => String(fragment.text || "").trim()).join(" ").replace(/\s+/g, " ").trim();
    const lineIndices = Array.from(new Set(fragments.map((fragment) => Number(fragment.line_index ?? 0)))).sort((a, b) => a - b);
    const bbox = [
      Math.min(...fragments.map((fragment) => Number(fragment.bbox?.[0] ?? 0))),
      Math.min(...fragments.map((fragment) => Number(fragment.bbox?.[1] ?? 0))),
      Math.max(...fragments.map((fragment) => Number(fragment.bbox?.[2] ?? 0))),
      Math.max(...fragments.map((fragment) => Number(fragment.bbox?.[3] ?? 0))),
    ];
    phrases.push({
      sentence_id: `${block?.unit_id || block?.id || "block"}:semantic_phrase:${sentenceIndex}`,
      sentence_index: sentenceIndex,
      text,
      texte: text,
      bbox,
      line_indices: lineIndices,
      start_line_index: lineIndices[0] ?? 0,
      end_line_index: lineIndices[lineIndices.length - 1] ?? 0,
      multi_line: lineIndices.length > 1,
      fragment_count: fragments.length,
      fragments,
      source: block?.source || "ocr",
      source_kind: "frontend_semantic_phrase",
      role: block?.role || "unknown",
      alignment: block?.alignment || "unknown",
      sentence_end_reason: endReason,
    });
    sentenceIndex += 1;
  };

  lines.forEach((line: any, index: number) => {
    const lineText = String(line?.line_text || line?.text || "").replace(/\s+/g, " ").trim();
    const previousLine = index > 0 ? lines[index - 1] : null;
    if (currentFragments.length && line?.hard_break_before && shouldBreakOnHardBoundary(currentText, line, previousLine)) {
      flush("hard_break_before");
    }
    if (!lineText) {
      return;
    }
    const chunks = splitLineTextIntoSentenceChunks(lineText);
    const normalizedChunks = chunks.length ? chunks : [{ text: lineText, start: 0, end: lineText.length, endsSentence: false }];
    normalizedChunks.forEach((chunk) => {
      currentText = appendSentenceFragment(currentText, chunk.text);
      currentFragments.push({
        fragment_index: currentFragments.length,
        line_index: Number(line?.line_index ?? index),
        text: chunk.text,
        bbox: approximateFragmentBBox(line?.bbox || [0, 0, 0, 0], lineText, chunk.start, chunk.end),
        source_line_text: lineText,
      });
      if (chunk.endsSentence) {
        flush("terminal_punctuation");
      }
    });
  });

  if (currentFragments.length) {
    flush("eof");
  }
  return phrases;
}

function getSemanticPhrasesForBlock(block: any) {
  const existing = Array.isArray(block?.semantic_phrases) ? block.semantic_phrases.filter((phrase: any) => Array.isArray(phrase?.bbox) && phrase.bbox.length === 4) : [];
  return existing.length ? existing : buildSemanticPhrasesFromBlock(block);
}

function getSemanticSpansForBlock(block: any) {
  const existing = Array.isArray(block?.semantic_spans) ? block.semantic_spans.filter((span: any) => Array.isArray(span?.bbox) && span.bbox.length === 4) : [];
  return existing;
}

function getExpressionSpansForPhrase(phrase: any) {
  const spans = Array.isArray(phrase?.spans) ? phrase.spans.filter((span: any) => Array.isArray(span?.bbox) && span.bbox.length === 4) : [];
  if (spans.length) {
    return spans;
  }
  const phraseText = String(phrase?.texte || phrase?.text || "").replace(/\s+/g, " ").trim();
  if (!phraseText || !Array.isArray(phrase?.bbox) || phrase.bbox.length !== 4) {
    return [];
  }
  return [
    {
      unit_id: `${phrase?.unit_id || "phrase"}:synthetic-span:0`,
      texte: phraseText,
      text: phraseText,
      bbox: phrase.bbox,
      source_kind: "synthetic_expression_span",
      translatable: phrase?.translatable,
      translation_strategy: phrase?.translation_strategy,
      text_attributes: phrase?.text_attributes,
      style_attributes: phrase?.style_attributes,
      layout_attributes: phrase?.layout_attributes,
      style: Array.isArray(phrase?.spans) && phrase.spans[0]?.style ? phrase.spans[0].style : {},
    },
  ];
}

function semanticClassForItem(item: InspectorItem) {
  const raw = item?.raw || {};
  return (
    raw?.expression_semantics?.inline_class ||
    raw?.editorial_semantics?.flow_class ||
    raw?.semantic?.role ||
    item?.role ||
    "unknown"
  );
}

function semanticColorForItem(item: InspectorItem) {
  const semanticClass = semanticClassForItem(item);
  const palette: Record<string, { stroke: string; fill: string; label: string }> = {
    editorial_body: { stroke: "#0d7bdc", fill: "rgba(64,155,255,0.08)", label: "Editorial body" },
    heading_like: { stroke: "#8a4fff", fill: "rgba(138,79,255,0.10)", label: "Heading" },
    anchored_annotation: { stroke: "#d35400", fill: "rgba(211,84,0,0.10)", label: "Annotation" },
    caption: { stroke: "#16a085", fill: "rgba(22,160,133,0.10)", label: "Caption" },
    reference_run: { stroke: "#6c757d", fill: "rgba(108,117,125,0.10)", label: "Reference" },
    protected_visual: { stroke: "#2c3e50", fill: "rgba(44,62,80,0.10)", label: "Protected" },
    technical_inline: { stroke: "#b35c00", fill: "rgba(179,92,0,0.10)", label: "Technical" },
    code: { stroke: "#1f7a1f", fill: "rgba(31,122,31,0.10)", label: "Code" },
    formula: { stroke: "#7f8c8d", fill: "rgba(127,140,141,0.10)", label: "Formula" },
    label: { stroke: "#c0392b", fill: "rgba(192,57,43,0.10)", label: "Label" },
    plain_text: { stroke: "#2980b9", fill: "rgba(41,128,185,0.08)", label: "Plain text" },
    unknown: { stroke: "#0d7bdc", fill: "rgba(64,155,255,0.08)", label: "Unknown" },
  };
  return palette[semanticClass] || palette.unknown;
}

function semanticLegendEntries(items: InspectorItem[]) {
  const entries = new Map<string, { stroke: string; fill: string; label: string; key: string }>();
  items.forEach((item) => {
    const key = semanticClassForItem(item);
    if (entries.has(key)) {
      return;
    }
    const color = semanticColorForItem(item);
    entries.set(key, { ...color, key });
  });
  return Array.from(entries.values());
}

function relationEndpointsForSelected(selectedItem: InspectorItem | null, items: InspectorItem[]) {
  if (!selectedItem) {
    return [] as RelationOverlay[];
  }
  const raw = selectedItem.raw || {};
  const relationCandidates = [
    { kind: "previous" as const, relation: raw?.expression_relations?.with_previous || raw?.editorial_relations?.with_previous },
    { kind: "next" as const, relation: raw?.expression_relations?.with_next || raw?.editorial_relations?.with_next },
  ];
  const currentBox = selectedItem.bbox || [0, 0, 0, 0];
  const cx = (Number(currentBox[0] || 0) + Number(currentBox[2] || 0)) / 2;
  const cy = (Number(currentBox[1] || 0) + Number(currentBox[3] || 0)) / 2;
  const overlays: RelationOverlay[] = [];
  relationCandidates.forEach(({ kind, relation }) => {
    if (!relation?.exists || !relation?.neighbor_id) {
      return;
    }
    const neighbor = items.find((item) => item.id === relation.neighbor_id || item.raw?.unit_id === relation.neighbor_id);
    if (!neighbor || !Array.isArray(neighbor.bbox) || neighbor.bbox.length !== 4) {
      return;
    }
    const nb = neighbor.bbox;
    const nx = (Number(nb[0] || 0) + Number(nb[2] || 0)) / 2;
    const ny = (Number(nb[1] || 0) + Number(nb[3] || 0)) / 2;
    overlays.push({
      key: `${selectedItem.id}-${kind}-${relation.neighbor_id}`,
      kind,
      x1: cx,
      y1: cy,
      x2: nx,
      y2: ny,
      color: kind === "previous" ? "#8a4fff" : "#16a085",
    });
  });
  return overlays;
}

function buildPhraseAuditRows(pageResult: any, translatedPageResult?: any | null) {
  return collectInspectorItems(pageResult, "phrase", translatedPageResult);
}

function csvEscape(value: any) {
  const text = String(value ?? "");
  if (/[",\n;]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function exportPhraseAuditCsv(rows: InspectorItem[]) {
  const header = ["Phrases", "Bloc", "Ligne", "Traduction", "Attributs"];
  const lines = [header.map(csvEscape).join(",")];
  rows.forEach((item) => {
    const raw = item.raw || {};
    const row = [
      item.text || "",
      raw?._parent_block_id || raw?.structural_context?.block_unit_id || "",
      formatLineRefs(raw?.line_indices),
      item.translatedText || "",
      flattenPhraseAttributes(raw, item.role, item.alignment),
    ];
    lines.push(row.map(csvEscape).join(","));
  });
  return lines.join("\n");
}

function collectInspectorItems(pageResult: any, level: ExtractionLevel, translatedPageResult?: any | null): InspectorItem[] {
  const structure = getStructureFromPageResult(pageResult);
  const blocks = structure?.blocks || [];
  const translatedStructure = getStructureFromPageResult(translatedPageResult);
  const translatedBlocksById = new Map<string, any>();
  (translatedStructure?.blocks || []).forEach((block: any, blockIndex: number) => {
    translatedBlocksById.set(blockKey(block, blockIndex), block);
  });
  const items: InspectorItem[] = [];

  blocks.forEach((block: any, blockIndex: number) => {
    if (!Array.isArray(block?.bbox) || block.bbox.length !== 4) {
      return;
    }
    const currentBlockId = blockKey(block, blockIndex);
    const translatedBlock = translatedBlocksById.get(currentBlockId) || null;
    const semanticPhrases = getSemanticPhrasesForBlock(block);
    const translatedSemanticPhrases = translatedBlock ? getSemanticPhrasesForBlock(translatedBlock) : [];
    const semanticSpans = getSemanticSpansForBlock(block);
    const translatedSemanticSpans = translatedBlock ? getSemanticSpansForBlock(translatedBlock) : [];
    if (level === "block") {
      items.push({
        id: currentBlockId,
        level,
        text: normalizeTextFromItem(block),
        translatedText: translatedBlock ? normalizeTextFromItem(translatedBlock) : "",
        role: block.role || "unknown",
        alignment: block.alignment || block.layout_attributes?.horizontal_alignment || "unknown",
        bbox: block.bbox,
        raw: block,
      });
      return;
    }

    if (level === "phrase" && Array.isArray(semanticPhrases) && semanticPhrases.length) {
      semanticPhrases.forEach((phrase: any, phraseIndex: number) => {
        if (!Array.isArray(phrase?.bbox) || phrase.bbox.length !== 4) {
          return;
        }
        items.push({
          id: phrase.unit_id || phrase.sentence_id || `${currentBlockId}-semantic-phrase-${phraseIndex}`,
          level,
          text: normalizeTextFromItem(phrase),
          translatedText: normalizeTextFromItem(translatedSemanticPhrases[phraseIndex]),
          role: phrase.role || block.role || "unknown",
          alignment: phrase.alignment || phrase.layout_attributes?.horizontal_alignment || block.alignment || "unknown",
          bbox: phrase.bbox,
          raw: {
            ...phrase,
            _parent_block_id: currentBlockId,
          },
        });
      });
      return;
    }

    if (level === "span" && Array.isArray(semanticSpans) && semanticSpans.length) {
      semanticSpans.forEach((span: any, spanIndex: number) => {
        if (!Array.isArray(span?.bbox) || span.bbox.length !== 4) {
          return;
        }
        items.push({
          id: span.unit_id || `${currentBlockId}-semantic-span-${spanIndex}`,
          level,
          text: normalizeTextFromItem(span),
          translatedText: normalizeTextFromItem(translatedSemanticSpans[spanIndex]),
          role: span.role || block.role || "unknown",
          alignment: span.alignment || span.layout_attributes?.horizontal_alignment || block.alignment || "unknown",
          bbox: span.bbox,
          raw: {
            ...span,
            _parent_block_id: currentBlockId,
          },
        });
      });
      return;
    }

    (block.lines || []).forEach((line: any, lineIndex: number) => {
      if (!Array.isArray(line?.bbox) || line.bbox.length !== 4) {
        return;
      }
      if (level === "line") {
        items.push({
          id: line.unit_id || `${currentBlockId}-line-${line.line_index ?? lineIndex}`,
          level,
          text: normalizeTextFromItem(line),
          translatedText: normalizeTextFromItem((translatedBlock?.lines || []).find((candidate: any) => Number(candidate?.line_index ?? -1) === Number(line?.line_index ?? lineIndex))),
          role: line.role || block.role || "unknown",
          alignment: line.alignment || line.layout_attributes?.horizontal_alignment || "unknown",
          bbox: line.bbox,
          raw: {
            ...line,
            _parent_block_id: currentBlockId,
            _parent_block_role: block.role || "unknown",
          },
        });
        return;
      }

      (line.phrases || []).forEach((phrase: any, phraseIndex: number) => {
        if (!Array.isArray(phrase?.bbox) || phrase.bbox.length !== 4) {
          return;
        }
        if (level === "phrase") {
          items.push({
            id:
              phrase.unit_id ||
              `${line.unit_id || `${blockIndex}-${lineIndex}`}-phrase-${phraseIndex}`,
          level,
          text: normalizeTextFromItem(phrase),
          translatedText: normalizeTextFromItem((((translatedBlock?.lines || []).find((candidate: any) => Number(candidate?.line_index ?? -1) === Number(line?.line_index ?? lineIndex))?.phrases) || [])[phraseIndex]),
          role: phrase.role || line.role || block.role || "unknown",
          alignment: phrase.alignment || phrase.layout_attributes?.horizontal_alignment || "unknown",
          bbox: phrase.bbox,
          raw: {
            ...phrase,
            _parent_line_id: line.unit_id || `line-${lineIndex}`,
            _parent_block_id: currentBlockId,
          },
        });
        return;
      }

        const translatedPhrase = (((translatedBlock?.lines || []).find((candidate: any) => Number(candidate?.line_index ?? -1) === Number(line?.line_index ?? lineIndex))?.phrases) || [])[phraseIndex];
        getExpressionSpansForPhrase(phrase).forEach((span: any, spanIndex: number) => {
          if (!Array.isArray(span?.bbox) || span.bbox.length !== 4) {
            return;
          }
          const translatedSpan = getExpressionSpansForPhrase(translatedPhrase || {})[spanIndex];
          items.push({
            id: span.unit_id || `${phrase.unit_id || `${blockIndex}-${lineIndex}-${phraseIndex}`}-span-${spanIndex}`,
            level,
            text: normalizeTextFromItem(span),
            translatedText: normalizeTextFromItem(translatedSpan) || normalizeTextFromItem(translatedPhrase),
            role: phrase.role || line.role || block.role || "unknown",
            alignment:
              span.alignment ||
              span.layout_attributes?.horizontal_alignment ||
              phrase.alignment ||
              phrase.layout_attributes?.horizontal_alignment ||
              "unknown",
            bbox: span.bbox,
            raw: {
              ...span,
              _parent_phrase_id: phrase.unit_id || `phrase-${phraseIndex}`,
              _parent_line_id: line.unit_id || `line-${lineIndex}`,
              _parent_block_id: currentBlockId,
            },
          });
        });
      });
    });
  });

  return items;
}

function ExtractionInspector({
  apiBaseUrl,
  ocrResult,
  translatedStructure,
  selectedPageIndex,
  onSelectPageIndex,
  level,
  onSelectLevel,
  query,
  onChangeQuery,
  selectedItemId,
  onSelectItemId,
}: {
  apiBaseUrl: string;
  ocrResult: any | null;
  translatedStructure: any | null;
  selectedPageIndex: number;
  onSelectPageIndex: (next: number) => void;
  level: ExtractionLevel;
  onSelectLevel: (next: ExtractionLevel) => void;
  query: string;
  onChangeQuery: (next: string) => void;
  selectedItemId: string | null;
  onSelectItemId: (next: string) => void;
}) {
  const pages = Array.isArray(ocrResult?.results) ? ocrResult.results : [];
  const translatedPages = getPageListFromStructurePayload(translatedStructure);
  const pageResult = pages[selectedPageIndex] || null;
  const translatedPageResult = translatedPages[selectedPageIndex] || null;
  const structure = getStructureFromPageResult(pageResult);
  const dimensions = structure?.dimensions || {};
  const pageWidth = Number(dimensions.width || 1);
  const pageHeight = Number(dimensions.height || 1);
  const sourceImageUrl = buildAssetUrl(
    apiBaseUrl,
    structure?.source_image_url || pageResult?.source_image_url || pageResult?.visual_url
  );
  const [semanticFilter, setSemanticFilter] = useState("all");
  const items = collectInspectorItems(pageResult, level, translatedPageResult);
  const phraseRows = buildPhraseAuditRows(pageResult, translatedPageResult);
  const legendEntries = semanticLegendEntries(items);
  const semanticFilterOptions = ["all", ...legendEntries.map((entry) => entry.key)];
  useEffect(() => {
    if (!semanticFilterOptions.includes(semanticFilter)) {
      setSemanticFilter("all");
    }
  }, [semanticFilter, semanticFilterOptions]);
  const queryNorm = query.trim().toLowerCase();
  const filteredItems = items.filter((item) => {
    if (semanticFilter !== "all" && semanticClassForItem(item) !== semanticFilter) {
      return false;
    }
    if (!queryNorm) {
      return true;
    }
    return JSON.stringify({
      text: item.text,
      translatedText: item.translatedText,
      role: item.role,
      alignment: item.alignment,
      raw: item.raw,
    })
      .toLowerCase()
      .includes(queryNorm);
  });
  const selectedItem =
    filteredItems.find((item) => item.id === selectedItemId) ||
    items.find((item) => item.id === selectedItemId) ||
    filteredItems[0] ||
    null;
  const relationOverlays = relationEndpointsForSelected(selectedItem, filteredItems);
  const filteredPhraseRows = phraseRows.filter((item) => {
    if (semanticFilter !== "all" && semanticClassForItem(item) !== semanticFilter) {
      return false;
    }
    if (!queryNorm) {
      return true;
    }
    return JSON.stringify({
      text: item.text,
      translatedText: item.translatedText,
      role: item.role,
      alignment: item.alignment,
      raw: item.raw,
    })
      .toLowerCase()
      .includes(queryNorm);
  });

  const handleExportPhraseCsv = () => {
    const csv = exportPhraseAuditCsv(filteredPhraseRows);
    if (Platform.OS === "web" && typeof Blob !== "undefined" && typeof URL !== "undefined") {
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const blobUrl = URL.createObjectURL(blob);
      openDownload(blobUrl, `inspection_phrases_page_${selectedPageIndex + 1}.csv`);
      setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
      return;
    }
    openDownload(`data:text/csv;charset=utf-8,${encodeURIComponent(csv)}`, `inspection_phrases_page_${selectedPageIndex + 1}.csv`);
  };

  return (
    <View style={styles.inspectorCard}>
      <View style={styles.previewHeader}>
        <View style={styles.previewHeaderCopy}>
          <Text style={styles.previewTitle}>Inspecteur d'extraction</Text>
          <Text style={styles.previewSubtitle}>
            OCR enrichi par page avec bboxes, attributs de layout, style et texte.
          </Text>
        </View>
        <StatusBadge label={level.toUpperCase()} tone="neutral" />
      </View>

      {!pages.length ? (
        <View style={styles.previewEmpty}>
          <Text style={styles.previewEmptyTitle}>Aucune extraction OCR</Text>
          <Text style={styles.previewEmptyBody}>Charge un document puis lance l'analyse OCR ou la traduction.</Text>
        </View>
      ) : (
        <>
          <View style={styles.inspectorToolbar}>
            <View style={styles.inspectorGroup}>
              <Text style={styles.optionLabel}>Pages</Text>
              <View style={styles.chipRow}>
                {pages.map((page: any, index: number) => {
                  const pageNumber = page?.page || index + 1;
                  const active = index === selectedPageIndex;
                  return (
                    <Pressable
                      key={`page-${pageNumber}-${index}`}
                      onPress={() => onSelectPageIndex(index)}
                      style={[styles.chip, active && styles.chipActive]}
                    >
                      <Text style={[styles.chipText, active && styles.chipTextActive]}>{`p.${pageNumber}`}</Text>
                    </Pressable>
                  );
                })}
              </View>
            </View>

            <View style={styles.inspectorGroup}>
              <Text style={styles.optionLabel}>Niveau</Text>
              <View style={styles.chipRow}>
                {EXTRACTION_LEVELS.map((option) => {
                  const active = option === level;
                  return (
                    <Pressable key={option} onPress={() => onSelectLevel(option)} style={[styles.chip, active && styles.chipActive]}>
                      <Text style={[styles.chipText, active && styles.chipTextActive]}>{option}</Text>
                    </Pressable>
                  );
                })}
              </View>
            </View>

            <View style={styles.inspectorSearchBox}>
              <Text style={styles.optionLabel}>Filtre</Text>
              <TextInput
                value={query}
                onChangeText={onChangeQuery}
                placeholder="texte, role, police, attribut..."
                placeholderTextColor="#8ea3b6"
                style={styles.backendInput}
              />
            </View>

            <View style={styles.inspectorGroup}>
              <Text style={styles.optionLabel}>Classe sémantique</Text>
              <View style={styles.chipRow}>
                {semanticFilterOptions.map((option) => {
                  const active = option === semanticFilter;
                  const label = option === "all" ? "toutes" : (legendEntries.find((entry) => entry.key === option)?.label || option);
                  return (
                    <Pressable key={option} onPress={() => setSemanticFilter(option)} style={[styles.chip, active && styles.chipActive]}>
                      <Text style={[styles.chipText, active && styles.chipTextActive]}>{label}</Text>
                    </Pressable>
                  );
                })}
              </View>
            </View>
          </View>

          <View style={styles.inspectorBody}>
            <View style={styles.inspectorCanvasCard}>
              <Text style={styles.canvasMeta}>
                {`${filteredItems.length}/${items.length} ${level}${filteredItems.length > 1 ? "s" : ""} visibles`}
              </Text>
              <View style={styles.legendWrap}>
                {legendEntries.map((entry) => (
                  <View key={entry.key} style={styles.legendItem}>
                    <View style={[styles.legendSwatch, { backgroundColor: entry.fill, borderColor: entry.stroke }]} />
                    <Text style={styles.legendText}>{entry.label}</Text>
                  </View>
                ))}
              </View>
              <View style={styles.inspectorCanvasViewport}>
                {Platform.OS === "web" && sourceImageUrl ? (
                  <View style={styles.webCanvasWrap as any}>
                    <img src={sourceImageUrl} alt="Source OCR" style={styles.webCanvasImage as any} />
                    <svg
                      viewBox={`0 0 ${pageWidth} ${pageHeight}`}
                      preserveAspectRatio="xMidYMid meet"
                      style={styles.webCanvasOverlay as any}
                    >
                      {filteredItems.map((item) => {
                        const [x0, y0, x1, y1] = item.bbox;
                        const isActive = selectedItem?.id === item.id;
                        const semanticColor = semanticColorForItem(item);
                        return (
                          <g key={item.id}>
                            <rect
                              x={x0}
                              y={y0}
                              width={Math.max(1, x1 - x0)}
                              height={Math.max(1, y1 - y0)}
                              fill={isActive ? "rgba(234,191,92,0.18)" : semanticColor.fill}
                              stroke={isActive ? "#e39a18" : semanticColor.stroke}
                              strokeWidth={isActive ? 3 : 1.4}
                              style={{ cursor: "pointer" }}
                              onClick={() => onSelectItemId(item.id)}
                            />
                          </g>
                        );
                      })}
                      <defs>
                        <marker id="arrow-prev" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                          <path d="M 0 0 L 10 5 L 0 10 z" fill="#8a4fff" />
                        </marker>
                        <marker id="arrow-next" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                          <path d="M 0 0 L 10 5 L 0 10 z" fill="#16a085" />
                        </marker>
                      </defs>
                      {relationOverlays.map((overlay) => (
                        <g key={overlay.key}>
                          <line
                            x1={overlay.x1}
                            y1={overlay.y1}
                            x2={overlay.x2}
                            y2={overlay.y2}
                            stroke={overlay.color}
                            strokeWidth={2.2}
                            strokeDasharray="6 4"
                            markerEnd={`url(#arrow-${overlay.kind})`}
                          />
                        </g>
                      ))}
                    </svg>
                  </View>
                ) : (
                  <View style={styles.previewEmpty}>
                    <Text style={styles.previewEmptyTitle}>Preview web requise</Text>
                    <Text style={styles.previewEmptyBody}>
                      Ouvre l'application en web pour superposer les bboxes directement sur l'image source.
                    </Text>
                  </View>
                )}
              </View>
            </View>

            <View style={styles.inspectorSidebar}>
              <View style={styles.inspectorListCard}>
                <Text style={styles.inspectorPanelTitle}>Extractions</Text>
                <ScrollView style={styles.inspectorListScroll}>
                  {filteredItems.map((item) => {
                    const active = selectedItem?.id === item.id;
                    return (
                      <Pressable
                        key={item.id}
                        onPress={() => onSelectItemId(item.id)}
                        style={[styles.inspectorListItem, active && styles.inspectorListItemActive]}
                      >
                        <Text style={styles.inspectorListMeta}>{`${item.role} · ${item.alignment}`}</Text>
                        <Text numberOfLines={3} style={styles.inspectorListText}>
                          {item.text || "[vide]"}
                        </Text>
                      </Pressable>
                    );
                  })}
                </ScrollView>
              </View>

              <View style={styles.inspectorDetailCard}>
                <Text style={styles.inspectorPanelTitle}>Attributs</Text>
                {selectedItem ? (
                  <ScrollView style={styles.inspectorDetailScroll}>
                    {(() => {
                      const ruleset = getRuleset(selectedItem.raw);
                      const rulesetSummary = getRulesetSummary(selectedItem.raw);
                      const relativeGeometry = selectedItem.raw?.relative_geometry || {};
                      const positioningPolicy = selectedItem.raw?.positioning_policy || {};
                      const elementRelations = selectedItem.raw?.element_relations || {};
                      const expressionSemantics = selectedItem.raw?.expression_semantics || {};
                      const expressionRelations = selectedItem.raw?.expression_relations || {};
                      const editorialSemantics = selectedItem.raw?.editorial_semantics || {};
                      const editorialRelations = selectedItem.raw?.editorial_relations || {};
                      const structuralContext = selectedItem.raw?.structural_context || {};
                      const semanticRuns = Array.isArray(selectedItem.raw?.semantic_runs) ? selectedItem.raw.semantic_runs : [];
                      const semanticGroups = Array.isArray(selectedItem.raw?.semantic_groups) ? selectedItem.raw.semantic_groups : [];
                      const semanticRole =
                        ruleset?.semantics?.role ||
                        rulesetSummary?.dominant_semantic_role ||
                        selectedItem.raw?.semantic?.role ||
                        editorialSemantics?.flow_class ||
                        expressionSemantics?.inline_class ||
                        "n/a";
                      return (
                        <>
                    <Text style={styles.contentBlockTitle}>Contenu</Text>
                    <Text style={styles.contentBlockValue}>{selectedItem.text || "[vide]"}</Text>
                    <Text style={styles.contentBlockTitle}>Traduction</Text>
                    <Text style={styles.contentBlockValue}>{selectedItem.translatedText || "non chargee"}</Text>

                    <AttributeSection
                      title="Identite"
                      rows={[
                        { label: "Niveau", value: selectedItem.level },
                        { label: "Role", value: selectedItem.role || "unknown" },
                        { label: "Alignement", value: selectedItem.alignment || "unknown" },
                        { label: "Classe couleur", value: semanticClassForItem(selectedItem) },
                        { label: "Identifiant", value: selectedItem.id },
                      ]}
                    />

                    <AttributeSection
                      title="Position"
                      rows={[
                        { label: "BBox", value: selectedItem.bbox },
                        { label: "Largeur", value: selectedItem.raw?.layout_attributes?.width_px },
                        { label: "Hauteur", value: selectedItem.raw?.layout_attributes?.height_px },
                        { label: "Anchor horizontal", value: selectedItem.raw?.layout_attributes?.horizontal_anchor },
                        { label: "Anchor vertical", value: selectedItem.raw?.layout_attributes?.vertical_anchor },
                        { label: "Alignement horizontal", value: selectedItem.raw?.layout_attributes?.horizontal_alignment },
                        { label: "Indentation px", value: selectedItem.raw?.layout_attributes?.indent_px },
                        { label: "Left gap px", value: selectedItem.raw?.layout_attributes?.left_gap_px },
                        { label: "Right gap px", value: selectedItem.raw?.layout_attributes?.right_gap_px },
                        { label: "Top gap px", value: selectedItem.raw?.layout_attributes?.top_gap_px },
                        { label: "Bottom gap px", value: selectedItem.raw?.layout_attributes?.bottom_gap_px },
                        { label: "Parent", value: relativeGeometry.parent_id },
                        { label: "Lecture", value: relativeGeometry.reading_order_index },
                        { label: "Chemin lecture", value: relativeGeometry.reading_order_path },
                        { label: "Colonne", value: relativeGeometry.column_id || relativeGeometry.container_block_id },
                        { label: "Debut ligne", value: selectedItem.raw?.start_line_index },
                        { label: "Fin ligne", value: selectedItem.raw?.end_line_index },
                        { label: "Multi-ligne", value: selectedItem.raw?.multi_line },
                      ]}
                    />

                    <AttributeSection
                      title="Style"
                      rows={[
                        { label: "Police dominante", value: selectedItem.raw?.style_attributes?.font_family_primary },
                        { label: "Taille mediane pt", value: selectedItem.raw?.style_attributes?.font_size_pt_median },
                        { label: "Taille min pt", value: selectedItem.raw?.style_attributes?.font_size_pt_min },
                        { label: "Taille max pt", value: selectedItem.raw?.style_attributes?.font_size_pt_max },
                        { label: "Couleur dominante", value: selectedItem.raw?.style_attributes?.color_primary },
                        { label: "Flags actifs", value: flagSummary(selectedItem.raw?.style_attributes?.flags_any) },
                        { label: "Bold ratio", value: selectedItem.raw?.style_attributes?.flags_ratio?.bold },
                        { label: "Italic ratio", value: selectedItem.raw?.style_attributes?.flags_ratio?.italic },
                        { label: "Uppercase ratio", value: selectedItem.raw?.style_attributes?.flags_ratio?.uppercase },
                        { label: "Monospace ratio", value: selectedItem.raw?.style_attributes?.flags_ratio?.monospace },
                        { label: "Serif ratio", value: selectedItem.raw?.style_attributes?.flags_ratio?.serif },
                      ]}
                    />

                    <AttributeSection
                      title="Texte"
                      rows={[
                        { label: "Nombre caracteres", value: selectedItem.raw?.text_attributes?.char_count },
                        { label: "Nombre mots", value: selectedItem.raw?.text_attributes?.word_count },
                        { label: "Nombre chiffres", value: selectedItem.raw?.text_attributes?.digit_count },
                        { label: "Ponctuation", value: selectedItem.raw?.text_attributes?.punctuation_count },
                        { label: "Profil de casse", value: selectedItem.raw?.text_attributes?.case_profile },
                        { label: "Ratio majuscules", value: selectedItem.raw?.text_attributes?.uppercase_ratio },
                        { label: "Ratio chiffres", value: selectedItem.raw?.text_attributes?.digit_ratio },
                        { label: "Ratio ponctuation", value: selectedItem.raw?.text_attributes?.punctuation_ratio },
                      ]}
                    />

                    <AttributeSection
                      title="Semantique"
                      rows={[
                        { label: "Role semantique", value: semanticRole },
                        { label: "Classe editoriale", value: editorialSemantics?.flow_class },
                        { label: "Reflowable", value: editorialSemantics?.reflowable },
                        { label: "Annote ancre", value: editorialSemantics?.anchored_annotation },
                        { label: "Heading like", value: editorialSemantics?.heading_like },
                        { label: "Caption like", value: editorialSemantics?.caption_like },
                        { label: "Protege visuel", value: editorialSemantics?.protected_visual },
                        { label: "Classe inline", value: expressionSemantics?.inline_class },
                        { label: "Inline protege", value: expressionSemantics?.protected_inline },
                        { label: "Inline immuable", value: expressionSemantics?.immutable_inline },
                        { label: "Inline technique", value: expressionSemantics?.technical_inline },
                        { label: "Niveau emphase", value: expressionSemantics?.emphasis_level },
                        { label: "Flags emphase", value: flagSummary(expressionSemantics?.emphasis_flags) },
                        { label: "Top semantic score", value: topScoreLabel(ruleset?.semantics?.role_scores || positioningPolicy?.semantic_context?.role_scores) },
                        { label: "Mode AI/heuristique", value: ruleset?.semantics?.model_used !== undefined ? (ruleset?.semantics?.model_used ? "modele" : "heuristique") : "n/a" },
                        { label: "Review ready", value: ruleset?.semantics?.review_ready },
                        { label: "Specialisation", value: ruleset?.semantics?.specialized_role_source },
                      ]}
                    />

                    <AttributeSection
                      title="Continuite"
                      rows={[
                        { label: "Avec precedent", value: summarizeContinuitySide(ruleset?.continuity?.with_previous) },
                        { label: "Avec suivant", value: summarizeContinuitySide(ruleset?.continuity?.with_next) },
                        { label: "Editorial precedent", value: editorialRelations?.with_previous?.exists ? `${editorialRelations?.with_previous?.relation}, gap ${formatScalar(editorialRelations?.with_previous?.gap_px)}px` : "aucune" },
                        { label: "Editorial suivant", value: editorialRelations?.with_next?.exists ? `${editorialRelations?.with_next?.relation}, gap ${formatScalar(editorialRelations?.with_next?.gap_px)}px` : "aucune" },
                        { label: "Expression precedent", value: expressionRelations?.with_previous?.exists ? `${expressionRelations?.with_previous?.relation}${expressionRelations?.with_previous?.same_style ? ", meme style" : ""}` : "aucune" },
                        { label: "Expression suivant", value: expressionRelations?.with_next?.exists ? `${expressionRelations?.with_next?.relation}${expressionRelations?.with_next?.same_style ? ", meme style" : ""}` : "aucune" },
                        { label: "Resume bloc", value: summarizePairRelations(elementRelations?.pair_relations) },
                        { label: "Flow from previous", value: positioningPolicy?.signals?.flow_from_previous },
                        { label: "Flow to next", value: positioningPolicy?.signals?.flow_to_next },
                        { label: "Keep with previous", value: ruleset?.rules?.keep_with_previous },
                        { label: "Keep with next", value: ruleset?.rules?.keep_with_next },
                        { label: "Continuity class", value: ruleset?.rules?.continuity_class },
                        { label: "Hard break before", value: ruleset?.rules?.hard_break_before },
                        { label: "Hard break after", value: ruleset?.rules?.hard_break_after },
                      ]}
                    />

                    <AttributeSection
                      title="Positionnement"
                      rows={[
                        { label: "Mode principal", value: positioningPolicy?.primary_position_reference?.mode || ruleset?.rules?.translation_positioning_mode },
                        { label: "Anchor H principal", value: positioningPolicy?.anchors?.horizontal?.primary || ruleset?.rules?.preserve_horizontal_anchor || rulesetSummary?.preferred_horizontal_anchor },
                        { label: "Anchor V principal", value: positioningPolicy?.anchors?.vertical?.primary || ruleset?.rules?.preserve_vertical_anchor || rulesetSummary?.preferred_vertical_anchor },
                        { label: "Anchor H secondaire", value: positioningPolicy?.anchors?.horizontal?.secondary || ruleset?.rules?.secondary_horizontal_anchor },
                        { label: "Anchor V secondaire", value: positioningPolicy?.anchors?.vertical?.secondary || ruleset?.rules?.secondary_vertical_anchor },
                        { label: "Croissance horizontale", value: positioningPolicy?.expansion_policy?.horizontal || ruleset?.rules?.horizontal_growth },
                        { label: "Croissance verticale", value: positioningPolicy?.expansion_policy?.vertical || ruleset?.rules?.vertical_growth },
                        { label: "Top score H", value: topScoreLabel(positioningPolicy?.anchors?.horizontal?.scores) },
                        { label: "Top score V", value: topScoreLabel(positioningPolicy?.anchors?.vertical?.scores) },
                        { label: "Reference combinee", value: topCombinedMode(ruleset?.position_reference_priority?.combined_modes) },
                      ]}
                    />

                    <AttributeSection
                      title="Contraintes"
                      rows={[
                        { label: "Horizontal reflow", value: ruleset?.constraints?.allow_horizontal_reflow },
                        { label: "Vertical reflow", value: ruleset?.constraints?.allow_vertical_reflow },
                        { label: "Preserve center", value: ruleset?.constraints?.preserve_center_if_possible },
                        { label: "Available left", value: ruleset?.constraints?.available_space?.left_px },
                        { label: "Available right", value: ruleset?.constraints?.available_space?.right_px },
                        { label: "Available top", value: ruleset?.constraints?.available_space?.top_px },
                        { label: "Available bottom", value: ruleset?.constraints?.available_space?.bottom_px },
                        { label: "Override conditions", value: ruleset?.override_conditions },
                      ]}
                    />

                    <AttributeSection
                      title="Extraction"
                      rows={[
                        { label: "Source", value: selectedItem.raw?.source },
                        { label: "Source kind", value: selectedItem.raw?.source_kind },
                        { label: "Parent direct", value: structuralContext?.parent_unit_id },
                        { label: "Bloc parent", value: structuralContext?.block_unit_id },
                        { label: "Ligne parente", value: structuralContext?.line_unit_id },
                        { label: "Nb lignes enfants", value: structuralContext?.child_line_count },
                        { label: "Nb phrases semantiques", value: structuralContext?.child_semantic_phrase_count },
                        { label: "Nb phrases enfants", value: structuralContext?.child_phrase_count },
                        { label: "Nb spans enfants", value: structuralContext?.child_span_count },
                        { label: "Nb runs semantiques", value: selectedItem.raw?.semantic_run_count ?? semanticRuns.length },
                        { label: "Run parent", value: selectedItem.raw?.parent_semantic_run_id },
                        { label: "Nb groupes semantiques", value: selectedItem.raw?.semantic_group_count ?? semanticGroups.length },
                        { label: "Groupe parent", value: selectedItem.raw?.parent_semantic_group_id },
                        { label: "Translatable", value: selectedItem.raw?.translatable },
                        { label: "Translation strategy", value: selectedItem.raw?.translation_strategy },
                        { label: "Reading order index", value: selectedItem.raw?.reading_order_index },
                        { label: "Line index", value: selectedItem.raw?.line_index },
                        { label: "Lignes source", value: selectedItem.raw?.line_indices },
                        { label: "Fragments", value: selectedItem.raw?.fragment_count },
                        { label: "Coverage required", value: selectedItem.raw?.coverage_required },
                      ]}
                    />
                        </>
                      );
                    })()}
                  </ScrollView>
                ) : (
                  <View style={styles.previewEmpty}>
                    <Text style={styles.previewEmptyTitle}>Aucune extraction selectionnee</Text>
                    <Text style={styles.previewEmptyBody}>Clique une bbox ou une entree de la liste.</Text>
                  </View>
                )}
              </View>
            </View>
          </View>

          <View style={styles.phraseTableCard}>
            <View style={styles.previewHeader}>
              <View style={styles.previewHeaderCopy}>
                <Text style={styles.previewTitle}>Tableau des phrases extraites</Text>
                <Text style={styles.previewSubtitle}>
                  Phrase complete, bloc parent, lignes source, traduction disponible et attributs cle:valeur.
                </Text>
              </View>
              <View style={styles.previewToolbar}>
                <StatusBadge label={`${filteredPhraseRows.length} phrases`} tone="neutral" />
                <Pressable style={styles.downloadButtonSecondary} onPress={handleExportPhraseCsv}>
                  <Text style={styles.downloadButtonSecondaryText}>Exporter CSV</Text>
                </Pressable>
              </View>
            </View>
            <ScrollView horizontal showsHorizontalScrollIndicator>
              <View style={styles.auditTable}>
                <View style={[styles.auditTableRow, styles.auditTableHeaderRow]}>
                  <Text style={[styles.auditTableCell, styles.auditCellPhrase, styles.auditTableHeaderText]}>Phrases</Text>
                  <Text style={[styles.auditTableCell, styles.auditCellBlock, styles.auditTableHeaderText]}>Bloc</Text>
                  <Text style={[styles.auditTableCell, styles.auditCellLine, styles.auditTableHeaderText]}>Ligne</Text>
                  <Text style={[styles.auditTableCell, styles.auditCellTranslation, styles.auditTableHeaderText]}>Traduction</Text>
                  <Text style={[styles.auditTableCell, styles.auditCellAttributes, styles.auditTableHeaderText]}>Attributs</Text>
                </View>
                <ScrollView style={styles.auditTableScroll}>
                  {filteredPhraseRows.map((item, index) => {
                    const active = selectedItem?.id === item.id;
                    const raw = item.raw || {};
                    return (
                      <Pressable
                        key={`audit-${item.id}`}
                        onPress={() => onSelectItemId(item.id)}
                        style={[
                          styles.auditTableRow,
                          index % 2 === 1 && styles.auditTableRowAlt,
                          active && styles.auditTableRowActive,
                        ]}
                      >
                        <Text style={[styles.auditTableCell, styles.auditCellPhrase, styles.auditTableText]}>
                          {item.text || "[vide]"}
                        </Text>
                        <Text style={[styles.auditTableCell, styles.auditCellBlock, styles.auditTableText]}>
                          {raw?._parent_block_id || "n/a"}
                        </Text>
                        <Text style={[styles.auditTableCell, styles.auditCellLine, styles.auditTableText]}>
                          {formatLineRefs(raw?.line_indices)}
                        </Text>
                        <Text style={[styles.auditTableCell, styles.auditCellTranslation, styles.auditTableText]}>
                          {item.translatedText || "non chargee"}
                        </Text>
                        <Text style={[styles.auditTableCell, styles.auditCellAttributes, styles.auditTableText]}>
                          {flattenPhraseAttributes(raw, item.role, item.alignment)}
                        </Text>
                      </Pressable>
                    );
                  })}
                </ScrollView>
              </View>
            </ScrollView>
          </View>
        </>
      )}
    </View>
  );
}

export default function App() {
  const { width } = useWindowDimensions();
  const isWide = width >= 1080;

  const [currentPage, setCurrentPage] = useState<AppPage>("translate");
  const [apiBaseUrl, setApiBaseUrl] = useState<string>(API_FALLBACK || "http://127.0.0.1:8001");
  const [connectionState, setConnectionState] = useState<ConnectionState>("detecting");
  const [connectionNote, setConnectionNote] = useState("Recherche automatique du backend...");
  const [targetLang, setTargetLang] = useState<TargetLang>("fr");
  const [style, setStyle] = useState<StyleOption>("professionnel");
  const [tone, setTone] = useState<ToneOption>("neutre");
  const [textRemovalMode, setTextRemovalMode] = useState<RemovalMode>("default");
  const [forceAi, setForceAi] = useState(false);
  const [fontAiAudit, setFontAiAudit] = useState(true);
  const [debugCompare, setDebugCompare] = useState(true);
  const [exportHtml, setExportHtml] = useState(true);
  const [selectedFile, setSelectedFile] = useState<PickedDocument | null>(null);
  const [status, setStatus] = useState("Choisis un document puis lance le pipeline.");
  const [busy, setBusy] = useState(false);
  const [ocrResult, setOcrResult] = useState<any | null>(null);
  const [translatedStructure, setTranslatedStructure] = useState<any | null>(null);
  const [reconstructResult, setReconstructResult] = useState<any | null>(null);
  const [translatedView, setTranslatedView] = useState<"html" | "pdf">("html");
  const [inspectorPageIndex, setInspectorPageIndex] = useState(0);
  const [inspectorLevel, setInspectorLevel] = useState<ExtractionLevel>("block");
  const [inspectorQuery, setInspectorQuery] = useState("");
  const [selectedInspectorItemId, setSelectedInspectorItemId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function detectBackend() {
      setConnectionState("detecting");
      setConnectionNote("Recherche automatique du backend...");
      try {
        const detected = await discoverBackendBaseUrl(apiBaseUrl);
        if (cancelled) {
          return;
        }
        setApiBaseUrl(detected.baseUrl);
        setConnectionState("connected");
        setConnectionNote(`Connecte a ${detected.baseUrl}`);
      } catch (error) {
        if (cancelled) {
          return;
        }
        setConnectionState("error");
        setConnectionNote(
          error instanceof Error
            ? error.message
            : "Backend introuvable. Demarre `python ocr_server.py`, puis recharge la page."
        );
      }
    }

    void detectBackend();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (reconstructResult?.pdf_url) {
      setTranslatedView("pdf");
    } else if (reconstructResult?.html_url) {
      setTranslatedView("html");
    }
  }, [reconstructResult?.pdf_url, reconstructResult?.html_url]);

  const ocrOptions: OcrOptions = {
    forceAi,
    fontAiAudit,
    textRemovalMode,
  };

  const reconstructOptions: ReconstructOptions = {
    targetLang,
    style,
    tone,
    debugCompare,
    exportHtml,
  };

  const handlePickFile = async () => {
    try {
      const file = await pickDocument();
      if (!file) {
        return;
      }
      setSelectedFile(file);
      setOcrResult(null);
      setTranslatedStructure(null);
      setReconstructResult(null);
      setInspectorPageIndex(0);
      setInspectorQuery("");
      setSelectedInspectorItemId(null);
      setStatus(`Document charge: ${file.name}`);
    } catch (error) {
      Alert.alert("Fichier", error instanceof Error ? error.message : "Erreur inconnue");
    }
  };

  const handleInspectExtraction = async () => {
    if (!selectedFile) {
      Alert.alert("Document manquant", "Choisis un PDF ou une image.");
      return;
    }
    setBusy(true);
    setStatus("Extraction OCR en cours...");
    try {
      const ocrPayload = await runOcrRequest(apiBaseUrl, selectedFile, ocrOptions);
      setOcrResult(ocrPayload);
      try {
        const translatedPayload = await runTranslateStructureRequest(apiBaseUrl, ocrPayload?.results ?? [], {
          targetLang,
          style,
          tone,
        });
        setTranslatedStructure(translatedPayload?.structure || translatedPayload || null);
      } catch {
        setTranslatedStructure(null);
      }
      setReconstructResult(null);
      setInspectorPageIndex(0);
      setInspectorQuery("");
      setSelectedInspectorItemId(null);
      setStatus("Extraction OCR disponible.");
    } catch (error) {
      setTranslatedStructure(null);
      setStatus("Echec de l'extraction OCR.");
      Alert.alert("OCR", error instanceof Error ? error.message : "Erreur inconnue");
    } finally {
      setBusy(false);
    }
  };

  const handleTranslateDocument = async () => {
    if (!selectedFile) {
      Alert.alert("Document manquant", "Choisis un PDF ou une image.");
      return;
    }
    setBusy(true);
    setStatus("Pipeline en cours...");
    try {
      const ocrPayload = await runOcrRequest(apiBaseUrl, selectedFile, ocrOptions);
      setOcrResult(ocrPayload);
      try {
        const translatedPayload = await runTranslateStructureRequest(apiBaseUrl, ocrPayload?.results ?? [], {
          targetLang,
          style,
          tone,
        });
        setTranslatedStructure(translatedPayload?.structure || translatedPayload || null);
      } catch {
        setTranslatedStructure(null);
      }
      setInspectorPageIndex(0);
      setInspectorQuery("");
      setSelectedInspectorItemId(null);
      const pages = ocrPayload?.results ?? [];
      const reconstructPayload = await runReconstructRequest(apiBaseUrl, pages, reconstructOptions);
      setReconstructResult(reconstructPayload);
      setStatus("Document traduit disponible.");
    } catch (error) {
      setTranslatedStructure(null);
      setStatus("Echec de la traduction.");
      Alert.alert("Traduction", error instanceof Error ? error.message : "Erreur inconnue");
    } finally {
      setBusy(false);
    }
  };

  const htmlUrl = reconstructResult?.html_url ? `${apiBaseUrl}${reconstructResult.html_url}` : null;
  const pdfUrl = reconstructResult?.pdf_url ? `${apiBaseUrl}${reconstructResult.pdf_url}` : null;
  const translatedPreviewUrl = translatedView === "html" && htmlUrl ? htmlUrl : pdfUrl;
  const translatedPreviewKind: PreviewKind = translatedView === "html" && htmlUrl ? "html" : "pdf";
  const sourcePreviewKind: PreviewKind = selectedFile?.mimeType?.startsWith("image/") ? "image" : "pdf";
  const connectionTone = connectionState === "connected" ? "success" : connectionState === "error" ? "warning" : "neutral";
  const pageTitle = currentPage === "translate" ? "Original / Traduction" : "Inspection d'extraction";
  const pageSubtitle =
    currentPage === "translate"
      ? "Charge un document, traduis-le et compare le rendu."
      : "Charge un document, lance l'OCR et inspecte les blocs, lignes, phrases et expressions.";

  return (
    <View style={styles.screen}>
      <StatusBar style="dark" />

      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.shell}>
          <View style={[styles.topBar, isWide && styles.topBarWide]}>
            <View style={styles.titleBlock}>
              <Text style={styles.eyebrow}>Docs Parser</Text>
              <Text style={styles.title}>{pageTitle}</Text>
              <Text style={styles.subtitle}>{pageSubtitle}</Text>
              <View style={styles.pageNavRow}>
                {APP_PAGES.map((page) => {
                  const active = currentPage === page;
                  const label = page === "translate" ? "Traduire" : "Inspecter";
                  return (
                    <Pressable
                      key={page}
                      onPress={() => setCurrentPage(page)}
                      style={[styles.pageNavChip, active && styles.pageNavChipActive]}
                    >
                      <Text style={[styles.pageNavChipText, active && styles.pageNavChipTextActive]}>{label}</Text>
                    </Pressable>
                  );
                })}
              </View>
            </View>
            <View style={styles.statusStack}>
              <StatusBadge
                label={connectionState === "connected" ? "Backend connecte" : connectionState === "error" ? "Backend indisponible" : "Connexion..."}
                tone={connectionTone}
              />
              <Text style={styles.connectionText}>{connectionNote}</Text>
            </View>
          </View>

          <View style={styles.controlCard}>
            <View style={[styles.controlTopRow, isWide && styles.controlTopRowWide]}>
              <View style={styles.documentMeta}>
                <Text style={styles.sectionLabel}>Document charge</Text>
                <Text style={styles.fileName}>{selectedFile ? selectedFile.name : "Aucun document selectionne"}</Text>
                <Text style={styles.backendText}>{apiBaseUrl}</Text>
              </View>
              <View style={styles.backendBox}>
                <Text style={styles.sectionLabel}>Backend</Text>
                <TextInput
                  value={apiBaseUrl}
                  onChangeText={setApiBaseUrl}
                  autoCapitalize="none"
                  autoCorrect={false}
                  placeholder="http://127.0.0.1:8001"
                  placeholderTextColor="#8ea3b6"
                  style={styles.backendInput}
                />
              </View>
              <View style={styles.actionCluster}>
                <Pressable disabled={busy} onPress={handlePickFile} style={[styles.secondaryButton, busy && styles.disabledButton]}>
                  <Text style={styles.secondaryButtonText}>Choisir</Text>
                </Pressable>
                {currentPage === "inspect" ? (
                  <Pressable
                    disabled={busy || connectionState !== "connected"}
                    onPress={handleInspectExtraction}
                    style={[styles.primaryButton, (busy || connectionState !== "connected") && styles.disabledButton]}
                  >
                    <Text style={styles.primaryButtonText}>Inspecter</Text>
                  </Pressable>
                ) : (
                  <Pressable
                    disabled={busy || connectionState !== "connected"}
                    onPress={handleTranslateDocument}
                    style={[styles.primaryButton, (busy || connectionState !== "connected") && styles.disabledButton]}
                  >
                    <Text style={styles.primaryButtonText}>Traduire</Text>
                  </Pressable>
                )}
              </View>
            </View>

            <View style={styles.optionGrid}>
              <ChipRow title="Langue" value={targetLang} options={TARGET_LANGS} onChange={setTargetLang} />
              <ChipRow title="Style" value={style} options={STYLES} onChange={setStyle} />
              <ChipRow title="Ton" value={tone} options={TONES} onChange={setTone} />
              <ChipRow title="Suppression texte" value={textRemovalMode} options={REMOVAL_MODES} onChange={setTextRemovalMode} />
            </View>

            <View style={styles.toggleGrid}>
              <ToggleRow label="Force AI extraction" value={forceAi} onChange={setForceAi} />
              <ToggleRow label="Audit AI des fontes" value={fontAiAudit} onChange={setFontAiAudit} />
              <ToggleRow label="Visual compare" value={debugCompare} onChange={setDebugCompare} />
              <ToggleRow label="Export HTML" value={exportHtml} onChange={setExportHtml} />
            </View>

            <View style={styles.runtimeRow}>
              {busy ? <ActivityIndicator color="#6ea98e" /> : <View style={styles.runtimeDot} />}
              <Text style={styles.runtimeText}>{status}</Text>
            </View>
          </View>

          {currentPage === "translate" ? (
            <View style={[styles.previewGrid, isWide && styles.previewGridWide]}>
              <PreviewPane
                title="Document charge"
                subtitle={selectedFile?.name || "Source locale"}
                src={selectedFile?.uri ?? null}
                kind={sourcePreviewKind}
                fallback="Aucun document charge"
              />

              <View style={styles.previewCard}>
                <View style={styles.previewHeader}>
                  <View style={styles.previewHeaderCopy}>
                    <Text style={styles.previewTitle}>Document traduit</Text>
                    <Text style={styles.previewSubtitle}>{translatedView === "html" && htmlUrl ? "Sortie HTML" : "Sortie PDF"}</Text>
                  </View>
                  <View style={styles.previewToolbar}>
                    {htmlUrl ? (
                      <Pressable onPress={() => setTranslatedView("html")} style={[styles.modeChip, translatedView === "html" && styles.modeChipActive]}>
                        <Text style={[styles.modeChipText, translatedView === "html" && styles.modeChipTextActive]}>HTML</Text>
                      </Pressable>
                    ) : null}
                    {pdfUrl ? (
                      <Pressable onPress={() => setTranslatedView("pdf")} style={[styles.modeChip, translatedView === "pdf" && styles.modeChipActive]}>
                        <Text style={[styles.modeChipText, translatedView === "pdf" && styles.modeChipTextActive]}>PDF</Text>
                      </Pressable>
                    ) : null}
                  </View>
                </View>

                <View style={styles.previewViewport}>
                  <PreviewSurface
                    title="Document traduit"
                    src={translatedPreviewUrl}
                    kind={translatedPreviewKind}
                    fallback="Aucun document traduit"
                  />
                </View>

                <View style={styles.downloadRow}>
                  {pdfUrl ? (
                    <Pressable style={styles.downloadButton} onPress={() => openDownload(pdfUrl, "reconstructed_output.pdf")}>
                      <Text style={styles.downloadButtonText}>Telecharger le PDF</Text>
                    </Pressable>
                  ) : null}
                  {htmlUrl ? (
                    <Pressable style={styles.downloadButtonSecondary} onPress={() => openDownload(htmlUrl, "reconstructed_output.html")}>
                      <Text style={styles.downloadButtonSecondaryText}>Telecharger le HTML</Text>
                    </Pressable>
                  ) : null}
                </View>
              </View>
            </View>
          ) : (
            <ExtractionInspector
              apiBaseUrl={apiBaseUrl}
              ocrResult={ocrResult}
              translatedStructure={translatedStructure}
              selectedPageIndex={inspectorPageIndex}
              onSelectPageIndex={(next) => {
                setInspectorPageIndex(next);
                setSelectedInspectorItemId(null);
              }}
              level={inspectorLevel}
              onSelectLevel={(next) => {
                setInspectorLevel(next);
                setSelectedInspectorItemId(null);
              }}
              query={inspectorQuery}
              onChangeQuery={setInspectorQuery}
              selectedItemId={selectedInspectorItemId}
              onSelectItemId={setSelectedInspectorItemId}
            />
          )}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#f4f7f5",
  },
  container: {
    paddingHorizontal: 16,
    paddingVertical: 16,
  },
  shell: {
    width: "100%",
    maxWidth: 1380,
    alignSelf: "center",
    gap: 12,
  },
  topBar: {
    backgroundColor: "#ffffff",
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(156, 188, 207, 0.22)",
    padding: 16,
    gap: 12,
  },
  topBarWide: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
  },
  titleBlock: {
    flex: 1,
    gap: 6,
  },
  eyebrow: {
    color: "#668e79",
    fontSize: 11,
    fontWeight: "700",
    letterSpacing: 1.8,
    textTransform: "uppercase",
  },
  title: {
    color: "#18354d",
    fontSize: 28,
    lineHeight: 34,
    fontWeight: "800",
    fontFamily: Platform.select({ ios: "Avenir Next", android: "serif", web: "Georgia, serif" }),
  },
  subtitle: {
    color: "#5d768b",
    fontSize: 15,
    lineHeight: 22,
  },
  pageNavRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 8,
  },
  pageNavChip: {
    borderRadius: 999,
    paddingHorizontal: 14,
    paddingVertical: 9,
    backgroundColor: "#edf5fb",
  },
  pageNavChipActive: {
    backgroundColor: "#8ed7b9",
  },
  pageNavChipText: {
    color: "#4a6982",
    fontSize: 12,
    fontWeight: "800",
  },
  pageNavChipTextActive: {
    color: "#17324a",
  },
  statusStack: {
    minWidth: 260,
    gap: 8,
    alignItems: "flex-start",
  },
  statusBadge: {
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 999,
    backgroundColor: "#eef5fb",
  },
  statusBadgeSuccess: {
    backgroundColor: "#e7f8ef",
  },
  statusBadgeWarning: {
    backgroundColor: "#fff4db",
  },
  statusBadgeText: {
    color: "#22405c",
    fontSize: 12,
    fontWeight: "700",
  },
  connectionText: {
    color: "#607b90",
    fontSize: 13,
    lineHeight: 18,
  },
  controlCard: {
    backgroundColor: "#ffffff",
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(156, 188, 207, 0.22)",
    padding: 16,
    gap: 12,
  },
  controlTopRow: {
    gap: 12,
  },
  controlTopRowWide: {
    flexDirection: "row",
    alignItems: "stretch",
  },
  documentMeta: {
    flex: 1.2,
    minWidth: 220,
    borderRadius: 8,
    backgroundColor: "#f8fbfd",
    padding: 14,
    gap: 4,
  },
  backendBox: {
    flex: 1,
    minWidth: 220,
    borderRadius: 8,
    backgroundColor: "#f8fbfd",
    padding: 14,
    gap: 8,
  },
  sectionLabel: {
    color: "#698397",
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 1.2,
  },
  fileName: {
    color: "#17324a",
    fontSize: 16,
    lineHeight: 22,
    fontWeight: "800",
  },
  backendText: {
    color: "#668198",
    fontSize: 13,
    lineHeight: 18,
  },
  backendInput: {
    borderWidth: 1,
    borderColor: "rgba(145, 179, 204, 0.35)",
    backgroundColor: "#ffffff",
    color: "#16324a",
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  actionCluster: {
    minWidth: 180,
    justifyContent: "space-between",
    gap: 10,
  },
  optionGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 14,
  },
  optionGroup: {
    flexGrow: 1,
    minWidth: 220,
    gap: 8,
  },
  optionLabel: {
    color: "#18354d",
    fontSize: 13,
    fontWeight: "700",
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  chip: {
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 9,
    backgroundColor: "#edf5fb",
  },
  chipActive: {
    backgroundColor: "#85cfb0",
  },
  chipText: {
    color: "#436681",
    fontSize: 12,
    fontWeight: "700",
  },
  chipTextActive: {
    color: "#17324a",
  },
  toggleGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  toggleRow: {
    minWidth: 240,
    flexGrow: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "rgba(145, 179, 204, 0.18)",
  },
  toggleLabel: {
    flex: 1,
    color: "#19364f",
    fontSize: 13,
    fontWeight: "700",
  },
  primaryButton: {
    backgroundColor: "#efc95d",
    borderRadius: 8,
    paddingHorizontal: 18,
    paddingVertical: 14,
    alignItems: "center",
    justifyContent: "center",
  },
  primaryButtonText: {
    color: "#18354d",
    fontSize: 14,
    fontWeight: "800",
  },
  secondaryButton: {
    backgroundColor: "#e8f4ff",
    borderRadius: 8,
    paddingHorizontal: 18,
    paddingVertical: 14,
    alignItems: "center",
    justifyContent: "center",
  },
  secondaryButtonText: {
    color: "#1d5670",
    fontSize: 14,
    fontWeight: "700",
  },
  disabledButton: {
    opacity: 0.55,
  },
  runtimeRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  runtimeDot: {
    width: 10,
    height: 10,
    borderRadius: 999,
    backgroundColor: "#76b79b",
  },
  runtimeText: {
    color: "#58748a",
    fontSize: 14,
    fontWeight: "600",
  },
  previewGrid: {
    gap: 14,
  },
  previewGridWide: {
    flexDirection: "row",
    alignItems: "stretch",
  },
  previewCard: {
    flex: 1,
    backgroundColor: "#ffffff",
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(156, 188, 207, 0.22)",
    padding: 16,
    gap: 12,
  },
  previewHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  previewHeaderCopy: {
    flex: 1,
    gap: 3,
  },
  previewTitle: {
    color: "#18354d",
    fontSize: 18,
    fontWeight: "800",
  },
  previewSubtitle: {
    color: "#688297",
    fontSize: 12,
    lineHeight: 17,
  },
  previewToolbar: {
    flexDirection: "row",
    gap: 8,
  },
  modeChip: {
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: "#edf5fb",
  },
  modeChipActive: {
    backgroundColor: "#8ed7b9",
  },
  modeChipText: {
    color: "#4a6982",
    fontWeight: "700",
    fontSize: 12,
  },
  modeChipTextActive: {
    color: "#17324a",
  },
  previewViewport: {
    minHeight: 760,
    flex: 1,
    backgroundColor: "#ffffff",
    borderRadius: 8,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(145, 179, 204, 0.22)",
  },
  previewEmpty: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
    backgroundColor: "#f6fbff",
    gap: 8,
  },
  previewEmptyTitle: {
    color: "#18354d",
    fontSize: 17,
    fontWeight: "800",
    textAlign: "center",
  },
  previewEmptyBody: {
    color: "#698397",
    fontSize: 14,
    lineHeight: 20,
    textAlign: "center",
    maxWidth: 320,
  },
  downloadRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  downloadButton: {
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 11,
    backgroundColor: "#6ea98e",
  },
  downloadButtonText: {
    color: "#ffffff",
    fontWeight: "800",
  },
  downloadButtonSecondary: {
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 11,
    backgroundColor: "#eef5fb",
  },
  downloadButtonSecondaryText: {
    color: "#1d5670",
    fontWeight: "800",
  },
  inspectorCard: {
    backgroundColor: "#ffffff",
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(156, 188, 207, 0.22)",
    padding: 16,
    gap: 14,
  },
  inspectorToolbar: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 14,
    alignItems: "flex-end",
  },
  inspectorGroup: {
    flexGrow: 1,
    minWidth: 240,
    gap: 8,
  },
  inspectorSearchBox: {
    flexGrow: 1.2,
    minWidth: 260,
    gap: 8,
  },
  inspectorBody: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 14,
  },
  inspectorCanvasCard: {
    flex: 1.5,
    minWidth: 480,
    gap: 10,
  },
  canvasMeta: {
    color: "#607b90",
    fontSize: 13,
    fontWeight: "700",
  },
  legendWrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  legendItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: "#f4f8fb",
  },
  legendSwatch: {
    width: 12,
    height: 12,
    borderRadius: 999,
    borderWidth: 1.5,
  },
  legendText: {
    color: "#47657d",
    fontSize: 12,
    fontWeight: "700",
  },
  inspectorCanvasViewport: {
    minHeight: 820,
    backgroundColor: "#ffffff",
    borderRadius: 8,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(145, 179, 204, 0.22)",
  },
  webCanvasWrap: {
    position: "relative",
    width: "100%",
    height: "100%",
    minHeight: 820,
    backgroundColor: "#f9fbfd",
  },
  webCanvasImage: {
    width: "100%",
    height: "100%",
    objectFit: "contain",
    backgroundColor: "#ffffff",
  },
  webCanvasOverlay: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
  },
  inspectorSidebar: {
    flex: 1,
    minWidth: 320,
    gap: 14,
  },
  inspectorListCard: {
    backgroundColor: "#f8fbfd",
    borderRadius: 8,
    padding: 12,
    gap: 10,
    maxHeight: 390,
    borderWidth: 1,
    borderColor: "rgba(145, 179, 204, 0.18)",
  },
  inspectorDetailCard: {
    backgroundColor: "#f8fbfd",
    borderRadius: 8,
    padding: 12,
    gap: 10,
    flex: 1,
    minHeight: 410,
    borderWidth: 1,
    borderColor: "rgba(145, 179, 204, 0.18)",
  },
  phraseTableCard: {
    backgroundColor: "#f8fbfd",
    borderRadius: 8,
    padding: 12,
    gap: 10,
    borderWidth: 1,
    borderColor: "rgba(145, 179, 204, 0.18)",
  },
  inspectorPanelTitle: {
    color: "#18354d",
    fontSize: 15,
    fontWeight: "800",
  },
  inspectorListScroll: {
    flexGrow: 0,
  },
  inspectorListItem: {
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "rgba(145, 179, 204, 0.18)",
    gap: 5,
    marginBottom: 8,
  },
  inspectorListItemActive: {
    borderColor: "#e39a18",
    backgroundColor: "#fff7e8",
  },
  inspectorListMeta: {
    color: "#6b8397",
    fontSize: 11,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.8,
  },
  inspectorListText: {
    color: "#17324a",
    fontSize: 13,
    lineHeight: 18,
  },
  inspectorDetailScroll: {
    flex: 1,
  },
  auditTable: {
    minWidth: 1480,
    borderRadius: 8,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(145, 179, 204, 0.18)",
    backgroundColor: "#ffffff",
  },
  auditTableScroll: {
    maxHeight: 380,
  },
  auditTableRow: {
    flexDirection: "row",
    alignItems: "stretch",
    borderTopWidth: 1,
    borderTopColor: "rgba(145, 179, 204, 0.14)",
    backgroundColor: "#ffffff",
  },
  auditTableHeaderRow: {
    borderTopWidth: 0,
    backgroundColor: "#eef5fb",
  },
  auditTableRowAlt: {
    backgroundColor: "#fbfdff",
  },
  auditTableRowActive: {
    backgroundColor: "#fff7e8",
  },
  auditTableCell: {
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRightWidth: 1,
    borderRightColor: "rgba(145, 179, 204, 0.14)",
  },
  auditCellPhrase: {
    width: 350,
  },
  auditCellBlock: {
    width: 180,
  },
  auditCellLine: {
    width: 120,
  },
  auditCellTranslation: {
    width: 350,
  },
  auditCellAttributes: {
    width: 480,
    borderRightWidth: 0,
  },
  auditTableHeaderText: {
    color: "#36546d",
    fontSize: 12,
    fontWeight: "800",
    textTransform: "uppercase",
    letterSpacing: 0.7,
  },
  auditTableText: {
    color: "#17324a",
    fontSize: 13,
    lineHeight: 18,
  },
  contentBlockTitle: {
    color: "#6b8397",
    fontSize: 11,
    fontWeight: "800",
    textTransform: "uppercase",
    letterSpacing: 0.9,
    marginBottom: 8,
  },
  contentBlockValue: {
    color: "#17324a",
    fontSize: 14,
    lineHeight: 21,
    backgroundColor: "#ffffff",
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(145, 179, 204, 0.18)",
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  attributeSection: {
    marginTop: 10,
    gap: 8,
  },
  attributeRow: {
    backgroundColor: "#ffffff",
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(145, 179, 204, 0.18)",
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 4,
  },
  attributeLabel: {
    color: "#698397",
    fontSize: 11,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.7,
  },
  attributeValue: {
    color: "#18354d",
    fontSize: 13,
    lineHeight: 18,
  },
  detailKey: {
    color: "#6b8397",
    fontSize: 11,
    fontWeight: "800",
    textTransform: "uppercase",
    letterSpacing: 0.9,
    marginTop: 10,
  },
  detailValue: {
    color: "#18354d",
    fontSize: 12,
    lineHeight: 18,
    fontFamily: Platform.select({ ios: "Menlo", android: "monospace", web: "ui-monospace, SFMono-Regular, Menlo, monospace" }),
  },
} as any);
