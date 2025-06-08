import React, { useRef, useEffect, useCallback, useState } from "react";
import { ConnectionStatus } from "../types";

interface LiveCameraProps {
  onVideoReady: (videoElement: HTMLVideoElement) => void;
  onVideoStop: () => void;
  connectionStatus: ConnectionStatus;
}

const LiveCamera: React.FC<LiveCameraProps> = ({
  onVideoReady,
  onVideoStop,
  connectionStatus,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamRef, setStreamRef] = useState<MediaStream | null>(null);
  const initializeCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setStreamRef(stream);

        const playPromise = videoRef.current.play();
        if (playPromise !== undefined) {
          playPromise
            .then(() => {
              if (videoRef.current) {
                setIsStreaming(true);
                onVideoReady(videoRef.current);
              }
            })
            .catch((err) => {
              console.error("Video playback was prevented:", err);
            });
        }
      }
    } catch (err) {
      console.error("Kamera açılamadı:", err);
    }
  }, [onVideoReady]);

  const stopCamera = useCallback(() => {
    if (streamRef) {
      streamRef.getTracks().forEach((track) => track.stop());
      setStreamRef(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    onVideoStop();
  }, [streamRef, onVideoStop]);

  const toggleCamera = useCallback(() => {
    if (isStreaming) {
      stopCamera();
    } else {
      initializeCamera();
    }
  }, [isStreaming, stopCamera, initializeCamera]);
  useEffect(() => {
    // Component unmount olduğunda kamerayı durdur
    return () => {
      if (streamRef) {
        streamRef.getTracks().forEach((track) => track.stop());
      }
    };
  }, [streamRef]);

  return (
    <div className="relative bg-gray-900 rounded-2xl overflow-hidden shadow-2xl border border-gray-200">
      {" "}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-white font-semibold flex items-center gap-2">
            🎥 Canlı Kamera
            <div
              className={`w-3 h-3 rounded-full ${
                connectionStatus === "connected"
                  ? "bg-green-400"
                  : connectionStatus === "connecting"
                  ? "bg-yellow-400"
                  : "bg-red-400"
              } animate-pulse`}
            ></div>
          </h3>

          <button
            onClick={toggleCamera}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              isStreaming
                ? "bg-red-500 hover:bg-red-600 text-white"
                : "bg-green-500 hover:bg-green-600 text-white"
            }`}
          >
            {isStreaming ? "🛑 Durdur" : "▶️ Başlat"}
          </button>
        </div>
      </div>{" "}
      <div className="aspect-video bg-black flex items-center justify-center relative">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          autoPlay
          muted
          playsInline
        />

        {!isStreaming && (
          <div className="absolute inset-0 bg-black/90 flex items-center justify-center">
            <div className="text-center p-6">
              <div className="text-blue-400 text-6xl mb-4">📷</div>
              <p className="text-white text-lg font-medium mb-2">
                Kamera Hazır
              </p>
              <p className="text-gray-300 text-sm">
                Analizi başlatmak için "Başlat" butonuna tıklayın
              </p>
            </div>
          </div>
        )}

        {connectionStatus === "error" && (
          <div className="absolute inset-0 bg-black/75 flex items-center justify-center">
            <div className="text-center p-6">
              <div className="text-red-400 text-6xl mb-4">📷</div>
              <p className="text-white text-lg font-medium">
                Kamera bağlantısı sağlanamadı
              </p>
              <p className="text-gray-300 text-sm mt-2">
                Lütfen kamera iznini kontrol edin
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LiveCamera;
