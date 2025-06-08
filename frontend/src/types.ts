export interface Speaker {
  id: number;
  is_speaking: boolean;
  speaking_time: number;
  emotion: string;
  emotion_confidence: number;
  emotion_history?: string[]; // Kişinin duygu geçmişi
  face_image?: ArrayBuffer | string;
}

export type ConnectionStatus =
  | "connecting"
  | "connected"
  | "disconnected"
  | "error";

export type AppMode = "live" | "upload" | "link";
