import { useEffect, useRef, useState, useCallback } from "react";
import { Speaker, ConnectionStatus } from "../types";

interface UseWebSocketReturn {
  connectionStatus: {
    status: ConnectionStatus;
    message: string;
  };
  speakers: Speaker[];
  sendMessage: (message: Record<string, unknown>) => void;
  sendBinaryData: (data: ArrayBuffer) => void;
  clearSpeakers: () => void;
}

export const useWebSocket = (url: string): UseWebSocketReturn => {
  const socketRef = useRef<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<{
    status: ConnectionStatus;
    message: string;
  }>({
    status: "connecting",
    message: "Bağlanıyor...",
  });
  const [speakers, setSpeakers] = useState<Speaker[]>([]);

  // Her kişi için duygu geçmişi tut
  const emotionHistoryRef = useRef<{ [speakerId: number]: string[] }>({});

  const sendMessage = useCallback((message: Record<string, unknown>) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message));
    }
  }, []);
  const sendBinaryData = useCallback((data: ArrayBuffer) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(data);
    }
  }, []);
  const clearSpeakers = useCallback(() => {
    setSpeakers([]);
    emotionHistoryRef.current = {}; // Duygu geçmişini de temizle
  }, []);

  useEffect(() => {
    console.log(`WebSocket bağlantısı başlatılıyor: ${url}`);

    try {
      socketRef.current = new WebSocket(url);

      socketRef.current.onopen = () => {
        console.log("WebSocket bağlantısı kuruldu");
        setConnectionStatus({
          status: "connected",
          message: "Bağlantı kuruldu",
        });
      };
      socketRef.current.onmessage = (event: MessageEvent) => {
        const data = JSON.parse(event.data);
        console.log("WebSocket mesajı alındı:", data);

        // Analiz verisi
        if (data.type === "analysisData" && data.data?.speakers) {
          // Her speaker için duygu geçmişini güncelle
          const updatedSpeakers = data.data.speakers.map((speaker: Speaker) => {
            // Mevcut duygu geçmişini al
            const currentHistory = emotionHistoryRef.current[speaker.id] || [];

            // Yeni duyguyu ekle (eğer farklıysa veya ilk kayıtsa)
            const lastEmotion = currentHistory[currentHistory.length - 1];
            if (!lastEmotion || lastEmotion !== speaker.emotion) {
              currentHistory.push(speaker.emotion);
              // En fazla 50 duygu tutmaya sınırla (performans için)
              if (currentHistory.length > 50) {
                currentHistory.shift();
              }
              emotionHistoryRef.current[speaker.id] = currentHistory;
            }

            // Speaker objesine emotion_history ekle
            return {
              ...speaker,
              emotion_history: [...currentHistory],
            };
          });

          setSpeakers(updatedSpeakers);
        }
        // Eski format için backward compatibility
        else if (data.speakers) {
          // Eski format için de aynı işlemi yap
          const updatedSpeakers = data.speakers.map((speaker: Speaker) => {
            const currentHistory = emotionHistoryRef.current[speaker.id] || [];
            const lastEmotion = currentHistory[currentHistory.length - 1];
            if (!lastEmotion || lastEmotion !== speaker.emotion) {
              currentHistory.push(speaker.emotion);
              if (currentHistory.length > 50) {
                currentHistory.shift();
              }
              emotionHistoryRef.current[speaker.id] = currentHistory;
            }
            return {
              ...speaker,
              emotion_history: [...currentHistory],
            };
          });

          setSpeakers(updatedSpeakers);
        }
        // YouTube stream eventi
        else if (data.event === "streamStarted") {
          console.log("YouTube stream başlatıldı:", data.streamId);
        } else if (data.event === "videoInfo") {
          console.log("Video bilgisi:", data.info);
        } else if (data.error) {
          console.error("Backend hatası:", data.error);
        }
      };

      socketRef.current.onclose = (event) => {
        console.log(
          `WebSocket bağlantısı kapandı. Kod: ${event.code}, Neden: ${event.reason}`
        );
        setConnectionStatus({
          status: "disconnected",
          message: `Bağlantı kapandı (${event.code})`,
        });
      };

      socketRef.current.onerror = (error) => {
        console.error("WebSocket hatası:", error);
        setConnectionStatus({
          status: "error",
          message: "Bağlantı hatası",
        });
      };

      return () => {
        if (socketRef.current) {
          socketRef.current.close();
        }
      };
    } catch (error) {
      console.error("WebSocket bağlantısı kurulurken hata oluştu:", error);
      setConnectionStatus({
        status: "error",
        message: "Bağlantı kurulamadı",
      });
    }
  }, [url]);
  return {
    connectionStatus,
    speakers,
    sendMessage,
    sendBinaryData,
    clearSpeakers,
  };
};
