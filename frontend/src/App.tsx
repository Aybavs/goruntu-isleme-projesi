import { useCallback, useEffect, useRef, useState } from "react";

import SpeakerCard from "./components/SpeakerCard";
import StatusBadge from "./components/StatusBadge";
import { Speaker } from "./types";

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<{
    status: "connecting" | "connected" | "disconnected" | "error";
    message: string;
  }>({
    status: "connecting",
    message: "Bağlanıyor...",
  });
  const [speakers, setSpeakers] = useState<Speaker[]>([]);

  const setupFrameCapture = useCallback(() => {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");

    const intervalId = setInterval(() => {
      if (
        videoRef.current &&
        context &&
        socketRef.current?.readyState === WebSocket.OPEN
      ) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

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

    // Return cleanup function
    return () => clearInterval(intervalId);
  }, []);

  const initializeCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        // Play with proper error handling
        const playPromise = videoRef.current.play();

        // Handle the play promise properly to avoid AbortError
        if (playPromise !== undefined) {
          return playPromise
            .then(() => {
              console.log("Video playback started successfully");
              return setupFrameCapture();
            })
            .catch((err) => {
              console.error("Video playback was prevented:", err);
              if (err.name !== "AbortError") {
                setConnectionStatus({
                  status: "error",
                  message: "Video oynatma başlatılamadı",
                });
              }
              return undefined;
            });
        }
      }
      return undefined;
    } catch (err) {
      console.error("Kamera açılamadı:", err);
      setConnectionStatus({
        status: "error",
        message: "Kamera erişimi sağlanamadı",
      });
      return undefined;
    }
  }, [setupFrameCapture]);

  useEffect(() => {
    const wsUrl = "ws://localhost:8080/ws";
    console.log(`WebSocket bağlantısı başlatılıyor: ${wsUrl}`);

    let cleanupFrameCapture: (() => void) | undefined;
    // Store the current value of videoRef to avoid stale references in cleanup
    const currentVideo = videoRef.current;

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

        // Vision Service'dan gelen yanıtları işleme
        if (data.speakers) {
          setSpeakers(data.speakers);
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

      // Fix the type error by properly handling the Promise
      initializeCamera()
        .then((cleanup) => {
          if (cleanup) {
            cleanupFrameCapture = cleanup;
          }
        })
        .catch((error) => {
          console.error("Camera initialization error:", error);
        });

      return () => {
        // Clean up the frame capture interval if it exists
        if (cleanupFrameCapture) {
          cleanupFrameCapture();
        }

        // Close the video tracks to properly release the camera
        // Use the captured reference instead of videoRef.current
        if (currentVideo && currentVideo.srcObject) {
          const tracks = (currentVideo.srcObject as MediaStream).getTracks();
          tracks.forEach((track) => track.stop());
        }

        // Close the WebSocket connection
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
  }, [initializeCamera]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Görüntü İşleme Analiz Sistemi
          </h1>
          <StatusBadge
            status={connectionStatus.status}
            message={connectionStatus.message}
          />
        </header>

        <div className="mb-8 flex justify-center">
          <div className="relative rounded-lg overflow-hidden shadow-xl border-4 border-white">
            <video
              ref={videoRef}
              className="w-full max-w-2xl bg-black"
              width="640"
              height="480"
              autoPlay
              muted
            />
            {connectionStatus.status === "error" && (
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                <p className="text-white text-lg font-medium px-4 py-2 bg-red-600 rounded-md">
                  Kamera bağlantısı sağlanamadı
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="speakers-section">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4 text-center">
            {speakers.length > 0
              ? `Tespit Edilen Kişiler (${speakers.length})`
              : "Hiç kimse tespit edilmedi"}
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {speakers.map((speaker) => (
              <SpeakerCard key={speaker.id} speaker={speaker} />
            ))}
          </div>

          {speakers.length === 0 && (
            <div className="text-center py-12">
              <p className="text-gray-500 text-lg">
                Henüz tespit edilmiş kişi bulunmuyor
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
