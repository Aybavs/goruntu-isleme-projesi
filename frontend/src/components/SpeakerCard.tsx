import { useMemo } from "react";

import { Speaker } from "../types";

interface SpeakerCardProps {
  speaker: Speaker & {
    dominantEmotion?: {
      emotion: string;
      count: number;
      totalCount: number;
      emoji: string;
      colorClass: string;
      percentage: number;
    };
  };
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

  // Duygu emoji'leri
  const emotionEmoji = (() => {
    switch (speaker.emotion.toLowerCase()) {
      case "happy":
        return "😊";
      case "sad":
        return "😢";
      case "angry":
        return "😠";
      case "surprised":
        return "😲";
      case "neutral":
        return "😐";
      default:
        return "❓";
    }
  })();

  // Konuşma süresini formatlama
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="bg-gradient-to-br from-white to-gray-50 rounded-xl shadow-lg border border-gray-200 p-4 hover:shadow-xl transition-all duration-300 hover:scale-105">
      <div className="flex flex-col space-y-3">
        {/* Kişi başlığı ve durum */}
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
            👤 Kişi {speaker.id}
          </h3>
          {speaker.is_speaking && (
            <div className="flex items-center space-x-1 text-green-600 bg-green-50 px-2 py-1 rounded-full">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-xs font-medium">Konuşuyor</span>
            </div>
          )}
        </div>
        {/* Yüz görüntüsü */}
        <div className="flex justify-center">
          {faceImageUrl ? (
            <img
              src={faceImageUrl}
              alt={`Kişi ${speaker.id}`}
              className="w-20 h-20 rounded-xl object-cover border-3 border-white shadow-md"
            />
          ) : (
            <div className="w-20 h-20 rounded-xl bg-gradient-to-br from-gray-300 to-gray-400 flex items-center justify-center border-3 border-white shadow-md">
              <span className="text-white text-xl font-bold">{speaker.id}</span>
            </div>
          )}
        </div>{" "}
        {/* Duygu durumu */}
        <div className="space-y-2">
          {/* Mevcut duygu */}
          <div className="text-center">
            <div className="text-xs text-gray-500 mb-1">Şu anki duygu</div>
            <span
              className={`inline-flex items-center px-3 py-2 rounded-xl text-sm font-semibold shadow-sm ${emotionColorClass}`}
            >
              {emotionEmoji} {speaker.emotion}
              <span className="ml-2 text-xs opacity-75 bg-white/30 px-2 py-1 rounded-full">
                %{Math.round(speaker.emotion_confidence * 100)}
              </span>
            </span>
          </div>

          {/* Baskın duygu */}
          {speaker.dominantEmotion && (
            <div className="text-center">
              <div className="text-xs text-gray-500 mb-1">Baskın duygu</div>
              <span
                className={`inline-flex items-center px-3 py-2 rounded-xl text-sm font-semibold shadow-sm ${speaker.dominantEmotion.colorClass}`}
              >
                {speaker.dominantEmotion.emoji}{" "}
                {speaker.dominantEmotion.emotion}
                <span className="ml-2 text-xs opacity-75 bg-white/30 px-2 py-1 rounded-full">
                  %{speaker.dominantEmotion.percentage}
                </span>
              </span>
              <div className="text-xs text-gray-400 mt-1">
                {speaker.dominantEmotion.count}/
                {speaker.dominantEmotion.totalCount} gözlem
              </div>
            </div>
          )}
        </div>
        {/* Konuşma süresi */}
        <div className="text-center bg-gray-50 rounded-lg p-2">
          <div className="text-xs text-gray-500 mb-1">Toplam Konuşma</div>
          <div className="text-lg font-bold text-gray-700">
            {formatTime(speaker.speaking_time)}
          </div>
        </div>
      </div>
    </div>
  );
}

export default SpeakerCard;
