import { useCallback } from "react";
import useFrameCapture from "./useFrameCapture";

interface UseVideoHandlersProps {
  sendMessage: (message: Record<string, unknown>) => void;
  sendBinaryData: (data: ArrayBuffer) => void;
}

interface UseVideoHandlersReturn {
  handleVideoLinkReady: (
    videoElement: HTMLVideoElement | null,
    originalUrl: string
  ) => void;
  handleVideoUploadReady: (videoElement: HTMLVideoElement) => () => void;
  handleVideoSelect: (file: File) => void;
  handleVideoStop: () => void;
}

export const useVideoHandlers = ({
  sendMessage,
  sendBinaryData,
}: UseVideoHandlersProps): UseVideoHandlersReturn => {
  const { setupFrameCapture } = useFrameCapture();

  const handleVideoLinkReady = useCallback(
    (videoElement: HTMLVideoElement | null, originalUrl: string) => {
      // YouTube URL'i varsa backend'e gönder
      if (
        originalUrl &&
        (originalUrl.includes("youtube.com") ||
          originalUrl.includes("youtu.be"))
      ) {
        console.log("YouTube analizi başlatılıyor:", originalUrl);
        sendMessage({
          type: "analyzeYoutube",
          youtubeUrl: originalUrl,
        });
      } else if (videoElement) {
        // Normal video için frame capture
        const cleanup = setupFrameCapture(videoElement, sendBinaryData);
        return cleanup;
      }
    },
    [setupFrameCapture, sendMessage, sendBinaryData]
  );

  const handleVideoUploadReady = useCallback(
    (videoElement: HTMLVideoElement) => {
      // Upload edilen videolar için frame capture
      const cleanup = setupFrameCapture(videoElement, sendBinaryData);
      return cleanup;
    },
    [setupFrameCapture, sendBinaryData]
  );
  const handleVideoSelect = useCallback((file: File) => {
    console.log("Video seçildi:", file.name);
    // Video yükleme işlemleri burada yapılacak
  }, []);

  const handleVideoStop = useCallback(() => {
    console.log("Video analizi durduruldu");
    // Video analizi durdurma işlemleri burada yapılacak
  }, []);

  return {
    handleVideoLinkReady,
    handleVideoUploadReady,
    handleVideoSelect,
    handleVideoStop,
  };
};
