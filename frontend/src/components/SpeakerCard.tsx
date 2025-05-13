import { useMemo } from 'react';

import { Speaker } from '../types';

interface SpeakerCardProps {
  speaker: Speaker;
}

function SpeakerCard({ speaker }: SpeakerCardProps) {
  // Base64 formatındaki görüntüyü data URL'e dönüştür
  const faceImageUrl = useMemo(() => {
    if (speaker.face_image) {
      // Eğer base64 string ise doğrudan URL'e çevir
      if (typeof speaker.face_image === "string") {
        return `data:image/jpeg;base64,${speaker.face_image}`;
      }

      // Eğer ArrayBuffer ise, önce base64'e çevir
      if (speaker.face_image instanceof ArrayBuffer) {
        const base64 = btoa(
          new Uint8Array(speaker.face_image).reduce(
            (data, byte) => data + String.fromCharCode(byte),
            ""
          )
        );
        return `data:image/jpeg;base64,${base64}`;
      }
    }
    return null;
  }, [speaker.face_image]);

  // Duygulara göre renk sınıfları
  const emotionColorClass = (() => {
    switch (speaker.emotion.toLowerCase()) {
      case "happy":
        return "bg-yellow-100 text-yellow-800";
      case "sad":
        return "bg-blue-100 text-blue-800";
      case "angry":
        return "bg-red-100 text-red-800";
      case "surprised":
        return "bg-purple-100 text-purple-800";
      case "neutral":
        return "bg-gray-100 text-gray-800";
      default:
        return "bg-gray-100 text-gray-600";
    }
  })();

  // Konuşma süresini formatlama
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-lg font-semibold text-gray-900">
            Kişi #{speaker.id}
          </span>
          {speaker.is_speaking ? (
            <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full animate-pulse">
              Konuşuyor
            </span>
          ) : (
            <span className="px-2 py-1 text-xs font-medium bg-gray-100 text-gray-800 rounded-full">
              Sessiz
            </span>
          )}
        </div>

        {faceImageUrl ? (
          <div className="mb-4 flex justify-center">
            <img
              src={faceImageUrl}
              alt={`Kişi ${speaker.id}`}
              className="w-32 h-32 object-cover rounded-lg border-2 border-gray-200"
            />
          </div>
        ) : (
          <div className="mb-4 flex justify-center">
            <div className="w-32 h-32 bg-gray-200 rounded-lg flex items-center justify-center">
              <span className="text-gray-400">Görüntü yok</span>
            </div>
          </div>
        )}

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-gray-500">Duygu:</span>
            <span
              className={`px-2 py-1 text-xs font-medium rounded-full ${emotionColorClass}`}
            >
              {speaker.emotion === "unknown" ? "Bilinmiyor" : speaker.emotion}
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-gray-500">Güven:</span>
            <span className="text-sm font-medium">
              {Math.round(speaker.emotion_confidence * 100)}%
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-gray-500">
              Konuşma Süresi:
            </span>
            <span className="text-sm font-medium">
              {formatTime(speaker.speaking_time)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SpeakerCard;
