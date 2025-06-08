import React from "react";
import { Speaker } from "../types";
import SpeakerCard from "./SpeakerCard";

interface DetectedPersonsPanelProps {
  speakers: Speaker[];
}

const DetectedPersonsPanel: React.FC<DetectedPersonsPanelProps> = ({
  speakers,
}) => {
  const sortedSpeakers = [...speakers].sort(
    (a, b) => b.speaking_time - a.speaking_time
  );

  return (
    <div className="bg-white rounded-2xl shadow-2xl border border-gray-200 h-full">
      <div className="bg-gradient-to-r from-green-600 to-green-700 p-4 rounded-t-2xl">
        {" "}
        <h3 className="text-white font-semibold flex items-center gap-2">
          👥 Tespit Edilen Kişiler
          <span className="bg-white/20 px-2 py-1 rounded-full text-sm">
            {sortedSpeakers.length}
          </span>
        </h3>
      </div>
      <div className="p-4 h-[calc(100%-4rem)] overflow-y-auto">
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
