import { useCallback, useEffect, useRef, useState } from "react";

import ModeSelector from "./components/ModeSelector";
import LiveCamera from "./components/LiveCamera";
import VideoUpload from "./components/VideoUpload";
import VideoLink from "./components/VideoLink";
import DetectedPersonsPanel from "./components/DetectedPersonsPanel";
import StatusBadge from "./components/StatusBadge";
import { Speaker } from "./types";

function App() {
  const [mode, setMode] = useState<"live" | "upload" | "link">("live");
  const [connectionStatus, setConnectionStatus] = useState<{
    status: "connecting" | "connected" | "disconnected" | "error";
    message: string;
  }>({
    status: "connecting",
    message: "Bağlanıyor...",
  });
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const socketRef = useRef<WebSocket | null>(null);

  const setupFrameCapture = useCallback((videoElement: HTMLVideoElement) => {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");

    const intervalId = setInterval(() => {
      if (
        videoElement &&
        context &&
        socketRef.current?.readyState === WebSocket.OPEN
      ) {
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(
          (blob) => {
            if (blob) {
              blob.arrayBuffer().then((buffer) => {
                socketRef.current?.send(buffer);
              });
            }
          },
          "image/jpeg",
          0.8
        );
      }
    }, 500);

    return () => clearInterval(intervalId);
  }, []);

  const handleVideoLinkReady = useCallback(
    (videoElement: HTMLVideoElement | null, originalUrl: string) => {
      // YouTube URL'i varsa backend'e gönder
      if (
        originalUrl &&
        (originalUrl.includes("youtube.com") ||
          originalUrl.includes("youtu.be"))
      ) {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
          console.log("YouTube analizi başlatılıyor:", originalUrl);
          socketRef.current.send(
            JSON.stringify({
              type: "analyzeYoutube",
              youtubeUrl: originalUrl,
            })
          );
        }
      } else if (videoElement) {
        // Normal video için frame capture
        const cleanup = setupFrameCapture(videoElement);
        return cleanup;
      }
    },
    [setupFrameCapture]
  );

  const handleVideoUploadReady = useCallback(
    (videoElement: HTMLVideoElement) => {
      // Upload edilen videolar için frame capture
      const cleanup = setupFrameCapture(videoElement);
      return cleanup;
    },
    [setupFrameCapture]
  );

  const handleVideoSelect = useCallback((file: File) => {
    console.log("Video seçildi:", file.name);
    // Video yükleme işlemleri burada yapılacak
  }, []);

  useEffect(() => {
    const wsUrl = "ws://localhost:8080/ws";
    console.log(`WebSocket bağlantısı başlatılıyor: ${wsUrl}`);

    try {
      socketRef.current = new WebSocket(wsUrl);

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
          setSpeakers(data.data.speakers);
        }
        // Eski format için backward compatibility
        else if (data.speakers) {
          setSpeakers(data.speakers);
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
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Görüntü İşleme Analiz Sistemi
              </h1>
              <p className="text-gray-600 mt-1">
                AI destekli yüz tanıma ve duygu analizi
              </p>
            </div>
            <StatusBadge
              status={connectionStatus.status}
              message={connectionStatus.message}
            />
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Mode Selector */}
        <div className="mb-8">
          <ModeSelector selectedMode={mode} onModeChange={setMode} />
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 h-[calc(100vh-280px)]">
          {/* Sol Panel - Kamera/Video/Link */}
          <div className="lg:col-span-2">
            {mode === "live" ? (
              <LiveCamera
                onVideoReady={handleVideoUploadReady}
                connectionStatus={connectionStatus.status}
              />
            ) : mode === "upload" ? (
              <VideoUpload onVideoSelect={handleVideoSelect} />
            ) : (
              <VideoLink onVideoLoad={handleVideoLinkReady} />
            )}
          </div>

          {/* Sağ Panel - Tespit Edilen Kişiler */}
          <div className="lg:col-span-1">
            <DetectedPersonsPanel speakers={speakers} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
