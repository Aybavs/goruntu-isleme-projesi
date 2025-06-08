import React from "react";
import { Speaker } from "../types";
import SpeakerCard from "./SpeakerCard";

interface DetectedPersonsPanelProps {
  speakers: Speaker[];
}

interface SpeakerWithDominantEmotion extends Speaker {
  dominantEmotion: {
    emotion: string;
    count: number;
    totalCount: number;
    emoji: string;
    colorClass: string;
    percentage: number;
  };
}

const DetectedPersonsPanel: React.FC<DetectedPersonsPanelProps> = ({
  speakers,
}) => {
  // Her kişi için baskın duyguyu hesapla
  const speakersWithDominantEmotion =
    React.useMemo((): SpeakerWithDominantEmotion[] => {
      return speakers.map((speaker) => {
        // Eğer speaker'ın emotion_history'si varsa onu kullan, yoksa mevcut emotion'ı kullan
        const emotionHistory = speaker.emotion_history || [speaker.emotion];

        // Her duygunun kaç kez görüldüğünü say
        const emotionCounts: { [key: string]: number } = {};
        emotionHistory.forEach((emotion: string) => {
          const emotionKey = emotion.toLowerCase();
          emotionCounts[emotionKey] = (emotionCounts[emotionKey] || 0) + 1;
        });

        // En çok görülen duyguyu bul
        const dominantEmotionKey = Object.keys(emotionCounts).reduce((a, b) =>
          emotionCounts[a] > emotionCounts[b] ? a : b
        );

        const getEmotionEmoji = (emotion: string) => {
          switch (emotion) {
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
        };

        const getEmotionColor = (emotion: string) => {
          switch (emotion) {
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
        };

        const totalEmotions = emotionHistory.length;
        const dominantCount = emotionCounts[dominantEmotionKey];
        const percentage = Math.round((dominantCount / totalEmotions) * 100);

        return {
          ...speaker,
          dominantEmotion: {
            emotion: dominantEmotionKey,
            count: dominantCount,
            totalCount: totalEmotions,
            emoji: getEmotionEmoji(dominantEmotionKey),
            colorClass: getEmotionColor(dominantEmotionKey),
            percentage: percentage,
          },
        };
      });
    }, [speakers]);
  // Konuşma sürelerine göre sırala (en çok konuşan önce)
  const sortedSpeakers = [...speakersWithDominantEmotion].sort(
    (a, b) => b.speaking_time - a.speaking_time
  );

  return (
    <div className="bg-white rounded-2xl shadow-2xl border border-gray-200 h-full">
      {" "}
      <div className="bg-gradient-to-r from-green-600 to-green-700 p-4 rounded-t-2xl">
        <div className="flex items-center justify-between">
          <h3 className="text-white font-semibold flex items-center gap-2">
            👥 Tespit Edilen Kişiler
            <span className="bg-white/20 px-2 py-1 rounded-full text-sm">
              {sortedSpeakers.length}
            </span>
          </h3>
        </div>
      </div>
      <div className="p-4 space-y-4 h-[calc(100%-4rem)] overflow-y-auto">
        {/* Kişiler Grid */}
        {sortedSpeakers.length > 0 ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {sortedSpeakers.map((speaker) => (
              <SpeakerCard key={speaker.id} speaker={speaker} />
            ))}
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="text-gray-300 text-8xl mb-4">👤</div>
            <h4 className="text-lg font-semibold text-gray-600 mb-2">
              Henüz kimse tespit edilmedi
            </h4>
            <p className="text-gray-500 text-sm">
              Video analizi başladığında tespit edilen kişiler burada görünecek
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DetectedPersonsPanel;
