// VideoAnalyzerC  // 1) Component mount'ında WebSocket'i aç
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8080/ws");
    ws.onopen = () => setStatus("connected");iner.tsx
import { useState, useEffect, useCallback } from "react";
import VideoLink from "./VideoLink";

type VisionResult = {
  // vision-service’den dönen cevabın shape’i
  // örn: landmarks, faces, speaker vs.
  landmarks: any;
  speech: any;
};

export default function VideoAnalyzerContainer() {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [results, setResults] = useState<VisionResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  // 1) Component mount’ında WebSocket’i aç
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:4000/your-ws-path");
    ws.onopen = () => setStatus("connected");
    ws.onerror = (e) => {
      console.error("WS error", e);
      setError("WebSocket bağlantı hatası");
    };
    ws.onclose = () => setStatus("closed");
    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data);
        if (data.event === "analysisStarted") {
          setStatus("analyzing");
          setResults([]);
        } else if (data.event === "visionResult") {
          // visionResult içinde speech de olabilir
          setResults((prev) => [...prev, data.data]);
        } else if (data.event === "analysisComplete") {
          setStatus("done");
        } else if (data.error) {
          setError(data.error);
          setStatus("error");
        }
      } catch {
        // binary frame olabilirse ignore
      }
    };
    setSocket(ws);
    return () => {
      ws.close();
    };
  }, []);

  // 2) VideoLink’ten gelen videoElement ile analiz isteği başlat
  const handleVideoLoad = useCallback(
    (videoEl: HTMLVideoElement) => {
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        setError("WebSocket bağlı değil");
        return;
      }
      // videoElement.src embed URL olduğu için kullanıcının verdiği orijinal linki
      // sakladığımız bir state’e alabiliriz. VideoLink komponentinden linki de
      // geri döndürmek daha basit:
      // onVideoLoad(videoEl, originalUrl)
      // Şimdilik videoEl.currentSrc’dan parse edelim:
      const url = videoEl.currentSrc;
      socket.send(JSON.stringify({ type: "analyzeYoutube", youtubeUrl: url }));
    },
    [socket]
  );

  return (
    <div className="p-4 space-y-6">
      {error && (
        <div className="p-3 bg-red-100 text-red-800 rounded">{error}</div>
      )}

      <VideoLink onVideoLoad={handleVideoLoad} />

      <div>
        <h4 className="font-semibold">
          Durum:{" "}
          <span className="font-normal">
            {status === "idle" && "Beklemede"}
            {status === "connected" && "Bağlandı"}
            {status === "analyzing" && "Analiz ediliyor…"}
            {status === "done" && "Tamamlandı"}
            {status === "error" && "Hata"}
            {status === "closed" && "Bağlantı kapandı"}
          </span>
        </h4>
      </div>

      {results.length > 0 && (
        <div className="space-y-4">
          <h4 className="font-semibold">Sonuçlar</h4>
          <ul className="list-disc pl-6">
            {results.map((r, idx) => (
              <li key={idx} className="mb-2">
                📷 Kare {idx + 1}:{" "}
                {r.landmarks
                  ? `Landmarks algılandı (${r.speech || "konuşma yok"})`
                  : "Yüz algılanmadı"}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
