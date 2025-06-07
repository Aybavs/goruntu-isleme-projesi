import React from "react";

interface ModeSelectorProps {
  selectedMode: "live" | "upload" | "link";
  onModeChange: (mode: "live" | "upload" | "link") => void;
}

const ModeSelector: React.FC<ModeSelectorProps> = ({
  selectedMode,
  onModeChange,
}) => {
  return (
    <div className="flex bg-white rounded-xl p-2 shadow-lg border border-gray-200 max-w-2xl mx-auto">
      <button
        onClick={() => onModeChange("live")}
        className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all duration-300 ${
          selectedMode === "live"
            ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-md transform scale-105"
            : "text-gray-600 hover:text-blue-600 hover:bg-blue-50"
        }`}
      >
        📹 Canlı Video
      </button>
      <button
        onClick={() => onModeChange("upload")}
        className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all duration-300 ${
          selectedMode === "upload"
            ? "bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-md transform scale-105"
            : "text-gray-600 hover:text-purple-600 hover:bg-purple-50"
        }`}
      >
        📁 Video Yükle
      </button>
      <button
        onClick={() => onModeChange("link")}
        className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all duration-300 ${
          selectedMode === "link"
            ? "bg-gradient-to-r from-green-600 to-green-700 text-white shadow-md transform scale-105"
            : "text-gray-600 hover:text-green-600 hover:bg-green-50"
        }`}
      >
        🔗 Video Link
      </button>
    </div>
  );
};

export default ModeSelector;
