import React, { useRef, useEffect, useCallback } from "react";
import { ConnectionStatus } from "../types";

interface LiveCameraProps {
  onVideoReady: (videoElement: HTMLVideoElement) => void;
  connectionStatus: ConnectionStatus;
}

const LiveCamera: React.FC<LiveCameraProps> = ({
  onVideoReady,
  connectionStatus,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);

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

        const playPromise = videoRef.current.play();
        if (playPromise !== undefined) {
          playPromise
            .then(() => {
              if (videoRef.current) {
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
  useEffect(() => {
    initializeCamera();

    // Current video element referansını capture edelim
    const currentVideo = videoRef.current;

    return () => {
      if (currentVideo && currentVideo.srcObject) {
        const tracks = (currentVideo.srcObject as MediaStream).getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, [initializeCamera]);

  return (
    <div className="relative bg-gray-900 rounded-2xl overflow-hidden shadow-2xl border border-gray-200">
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 p-4">
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
      </div>

      <div className="aspect-video bg-black flex items-center justify-center">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          autoPlay
          muted
          playsInline
        />

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
