import { useCallback } from "react";

interface UseFrameCaptureReturn {
  setupFrameCapture: (
    videoElement: HTMLVideoElement,
    onFrame: (buffer: ArrayBuffer) => void
  ) => () => void;
}

const useFrameCapture = (): UseFrameCaptureReturn => {
  const setupFrameCapture = useCallback(
    (
      videoElement: HTMLVideoElement,
      onFrame: (buffer: ArrayBuffer) => void
    ) => {
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");

      const intervalId = setInterval(() => {
        if (videoElement && context) {
          canvas.width = videoElement.videoWidth;
          canvas.height = videoElement.videoHeight;
          context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

          canvas.toBlob(
            (blob) => {
              if (blob) {
                blob.arrayBuffer().then((buffer) => {
                  onFrame(buffer);
                });
              }
            },
            "image/jpeg",
            0.8
          );
        }
      }, 500);

      return () => clearInterval(intervalId);
    },
    []
  );

  return { setupFrameCapture };
};

export default useFrameCapture;
