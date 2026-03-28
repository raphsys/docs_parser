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
import { LinearGradient } from "expo-linear-gradient";
import { StatusBar } from "expo-status-bar";

import { pickDocument, type PickedDocument } from "./lib/filePicker";
import {
  discoverBackendBaseUrl,
  runOcrRequest,
  runReconstructRequest,
  type OcrOptions,
  type ReconstructOptions,
} from "./lib/api";

type PreviewKind = "pdf" | "html" | "image";
type ConnectionState = "detecting" | "connected" | "error";
type TargetLang = "fr" | "en" | "de" | "es";
type ToneOption = "neutre" | "didactique" | "analytique" | "formel";
type StyleOption = "professionnel" | "technique" | "scientifique";
type RemovalMode = "default" | "telea" | "ns";

const API_FALLBACK = Platform.select({
  web: "http://127.0.0.1:8001",
  default: "http://10.0.2.2:8001",
});

const TARGET_LANGS: TargetLang[] = ["fr", "en", "de", "es"];
const STYLES: StyleOption[] = ["professionnel", "technique", "scientifique"];
const TONES: ToneOption[] = ["neutre", "didactique", "analytique", "formel"];
const REMOVAL_MODES: RemovalMode[] = ["default", "telea", "ns"];

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

export default function App() {
  const { width } = useWindowDimensions();
  const isWide = width >= 1080;

  const [apiBaseUrl, setApiBaseUrl] = useState(API_FALLBACK ?? "http://127.0.0.1:8001");
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
  const [reconstructResult, setReconstructResult] = useState<any | null>(null);
  const [translatedView, setTranslatedView] = useState<"html" | "pdf">("html");

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
      setReconstructResult(null);
      setStatus(`Document charge: ${file.name}`);
    } catch (error) {
      Alert.alert("Fichier", error instanceof Error ? error.message : "Erreur inconnue");
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
      const pages = ocrPayload?.results ?? [];
      const reconstructPayload = await runReconstructRequest(apiBaseUrl, pages, reconstructOptions);
      setReconstructResult(reconstructPayload);
      setStatus("Document traduit disponible.");
    } catch (error) {
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

  return (
    <LinearGradient colors={["#fdfcf7", "#f4fbf8", "#eef6ff"]} style={styles.screen}>
      <StatusBar style="dark" />
      <View style={styles.bgCircleTop} />
      <View style={styles.bgCircleBottom} />

      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.shell}>
          <View style={[styles.topBar, isWide && styles.topBarWide]}>
            <View style={styles.titleBlock}>
              <Text style={styles.eyebrow}>Docs Parser</Text>
              <Text style={styles.title}>Original / Traduction</Text>
              <Text style={styles.subtitle}>Options en haut, comparaison des documents en dessous.</Text>
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
                <Pressable
                  disabled={busy || connectionState !== "connected"}
                  onPress={handleTranslateDocument}
                  style={[styles.primaryButton, (busy || connectionState !== "connected") && styles.disabledButton]}
                >
                  <Text style={styles.primaryButtonText}>Traduire</Text>
                </Pressable>
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
        </View>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#f7faf8",
  },
  bgCircleTop: {
    position: "absolute",
    top: -120,
    left: -80,
    width: 300,
    height: 300,
    borderRadius: 999,
    backgroundColor: "rgba(122, 190, 164, 0.18)",
  },
  bgCircleBottom: {
    position: "absolute",
    right: -110,
    bottom: -150,
    width: 360,
    height: 360,
    borderRadius: 999,
    backgroundColor: "rgba(234, 191, 92, 0.16)",
  },
  container: {
    paddingHorizontal: 16,
    paddingVertical: 18,
  },
  shell: {
    width: "100%",
    maxWidth: 1380,
    alignSelf: "center",
    gap: 14,
  },
  topBar: {
    backgroundColor: "rgba(255,255,255,0.9)",
    borderRadius: 24,
    borderWidth: 1,
    borderColor: "rgba(156, 188, 207, 0.22)",
    padding: 20,
    gap: 14,
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
    fontSize: 30,
    lineHeight: 36,
    fontWeight: "800",
    fontFamily: Platform.select({ ios: "Avenir Next", android: "serif", web: "Georgia, serif" }),
  },
  subtitle: {
    color: "#5d768b",
    fontSize: 15,
    lineHeight: 22,
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
    backgroundColor: "rgba(255,255,255,0.92)",
    borderRadius: 24,
    borderWidth: 1,
    borderColor: "rgba(156, 188, 207, 0.22)",
    padding: 18,
    gap: 14,
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
    borderRadius: 18,
    backgroundColor: "#f8fbfd",
    padding: 14,
    gap: 4,
  },
  backendBox: {
    flex: 1,
    minWidth: 220,
    borderRadius: 18,
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
    borderRadius: 14,
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
    borderRadius: 16,
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
    borderRadius: 18,
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
    borderRadius: 18,
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
    backgroundColor: "rgba(255,255,255,0.94)",
    borderRadius: 24,
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
    borderRadius: 20,
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
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 11,
    backgroundColor: "#6ea98e",
  },
  downloadButtonText: {
    color: "#ffffff",
    fontWeight: "800",
  },
  downloadButtonSecondary: {
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 11,
    backgroundColor: "#eef5fb",
  },
  downloadButtonSecondaryText: {
    color: "#1d5670",
    fontWeight: "800",
  },
});
