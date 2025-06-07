import React, { useState, useRef, useCallback } from "react";

interface VideoLinkProps {
  onVideoLoad: (
    videoElement: HTMLVideoElement | null,
    originalUrl: string
  ) => void;
}

const VideoLink: React.FC<VideoLinkProps> = ({ onVideoLoad }) => {
  const [url, setUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isVideoLoaded, setIsVideoLoaded] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const validateUrl = (url: string): boolean => {
    try {
      const urlObj = new URL(url);
      // YouTube, Vimeo, direct video files gibi desteklenen formatları kontrol et
      const supportedDomains = ["youtube.com", "youtu.be", "vimeo.com"];
      const supportedExtensions = [".mp4", ".webm", ".ogg", ".avi", ".mov"];

      const isDomainSupported = supportedDomains.some((domain) =>
        urlObj.hostname.includes(domain)
      );
      const isDirectVideo = supportedExtensions.some((ext) =>
        urlObj.pathname.toLowerCase().includes(ext)
      );

      return (
        isDomainSupported ||
        isDirectVideo ||
        urlObj.protocol === "http:" ||
        urlObj.protocol === "https:"
      );
    } catch {
      return false;
    }
  };
  const handleLoadVideo = useCallback(async () => {
    if (!url.trim()) {
      setError("Lütfen bir video linki girin");
      return;
    }

    if (!validateUrl(url)) {
      setError("Geçersiz URL formatı");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // YouTube URL'leri için embed format'a çevirmeyip direkt backend'e yolla
      if (url.includes("youtube.com") || url.includes("youtu.be")) {
        // YouTube URL'i için direkt analiz başlat
        setIsLoading(false);
        setIsVideoLoaded(true);
        onVideoLoad(null, url); // video element gereksiz, sadece URL gönder
        return;
      }

      // Diğer video türleri için normal yükleme
      if (videoRef.current) {
        videoRef.current.src = url;
        videoRef.current.onloadeddata = () => {
          setIsLoading(false);
          setIsVideoLoaded(true);
          if (videoRef.current) {
            onVideoLoad(videoRef.current, url);
          }
        };

        videoRef.current.onerror = () => {
          setIsLoading(false);
          setError("Video yüklenemedi. URL'yi kontrol edin.");
        };

        await videoRef.current.load();
      }
    } catch (err) {
      setIsLoading(false);
      setError("Video yüklenirken bir hata oluştu");
      console.error("Video load error:", err);
    }
  }, [url, onVideoLoad]);

  const handleReset = () => {
    setUrl("");
    setIsVideoLoaded(false);
    setError(null);
    if (videoRef.current) {
      videoRef.current.src = "";
    }
  };

  return (
    <div className="bg-white rounded-2xl overflow-hidden shadow-2xl border border-gray-200">
      <div className="bg-gradient-to-r from-green-600 to-green-700 p-4">
        <h3 className="text-white font-semibold flex items-center gap-2">
          🔗 Video Link
        </h3>
      </div>

      <div className="p-6">
        {!isVideoLoaded ? (
          <div className="space-y-4">
            <div>
              <label
                htmlFor="video-url"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Video URL'si
              </label>
              <input
                id="video-url"
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/video.mp4 veya YouTube linki"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-colors"
                disabled={isLoading}
              />
            </div>

            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-600 text-sm flex items-center gap-2">
                  ❌ {error}
                </p>
              </div>
            )}

            <div className="text-center">
              <button
                onClick={handleLoadVideo}
                disabled={isLoading || !url.trim()}
                className="bg-gradient-to-r from-green-600 to-green-700 text-white px-6 py-3 rounded-lg font-semibold hover:from-green-700 hover:to-green-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl"
              >
                {isLoading ? (
                  <span className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Yükleniyor...
                  </span>
                ) : (
                  "Video Yükle"
                )}
              </button>
            </div>

            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-800 mb-2">
                📝 Desteklenen Formatlar:
              </h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• YouTube videoları (youtube.com, youtu.be)</li>
                <li>• Doğrudan video linkleri (.mp4, .webm, .ogg)</li>
                <li>• Vimeo videoları</li>
                <li>• Diğer HTTPS video linkleri</li>
              </ul>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {" "}
            {/* YouTube URL'leri için özel görünüm */}
            {url.includes("youtube.com") || url.includes("youtu.be") ? (
              <div className="aspect-video bg-gradient-to-br from-red-50 to-red-100 rounded-lg overflow-hidden border-2 border-red-200 flex items-center justify-center">
                <div className="text-center p-6">
                  <div className="text-red-500 text-6xl mb-4">🎬</div>
                  <h4 className="text-lg font-semibold text-red-700 mb-2">
                    YouTube Analizi Başlatıldı
                  </h4>
                  <p className="text-red-600 mb-4">
                    Video backend'de işleniyor...
                  </p>
                  <div className="flex justify-center mb-4">
                    <div className="w-8 h-8 border-4 border-red-300 border-t-red-600 rounded-full animate-spin"></div>
                  </div>
                  <div className="bg-white rounded-lg p-3 shadow-sm">
                    <p className="text-sm text-gray-600 break-all">{url}</p>
                  </div>
                </div>
              </div>
            ) : (
              // Normal videolar için video oynatıcı
              <div className="aspect-video bg-black rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover"
                  controls
                  playsInline
                />
              </div>
            )}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-green-600">
                <span className="text-lg">✅</span>
                <span className="font-medium">
                  {url.includes("youtube.com") || url.includes("youtu.be")
                    ? "YouTube analizi başlatıldı"
                    : "Video başarıyla yüklendi"}
                </span>
              </div>
              <button
                onClick={handleReset}
                className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
              >
                {url.includes("youtube.com") || url.includes("youtu.be")
                  ? "Yeni Video"
                  : "Yeni Video"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoLink;
