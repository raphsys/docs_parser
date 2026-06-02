import * as DocumentPicker from "expo-document-picker";
import { Platform } from "react-native";

export type PickedDocument = {
  uri: string;
  name: string;
  mimeType: string;
  webFile?: File;
};

function pickWebDocument(): Promise<PickedDocument | null> {
  return new Promise((resolve, reject) => {
    if (typeof document === "undefined") {
      reject(new Error("Web file input is unavailable."));
      return;
    }
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".pdf,image/*";
    input.onchange = () => {
      const file = input.files?.[0];
      if (!file) {
        resolve(null);
        return;
      }
      resolve({
        uri: URL.createObjectURL(file),
        name: file.name,
        mimeType: file.type || "application/octet-stream",
        webFile: file,
      });
    };
    input.onerror = () => reject(new Error("File selection failed."));
    input.click();
  });
}

export async function pickDocument(): Promise<PickedDocument | null> {
  if (Platform.OS === "web") {
    return pickWebDocument();
  }
  const result = await DocumentPicker.getDocumentAsync({
    copyToCacheDirectory: true,
    multiple: false,
    type: ["application/pdf", "image/*"],
  });
  if (result.type === "cancel") {
    return null;
  }
  return {
    uri: result.uri,
    name: result.name,
    mimeType: result.mimeType || "application/octet-stream",
  };
}
