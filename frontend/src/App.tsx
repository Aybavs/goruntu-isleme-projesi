import { useState, useEffect } from "react";

import ModeSelector from "./components/ModeSelector";
import AppHeader from "./components/AppHeader";
import MainContent from "./components/MainContent";
import DetectedPersonsPanel from "./components/DetectedPersonsPanel";
import { AppMode } from "./types";
import { useWebSocket } from "./hooks/useWebSocket";
import { useVideoHandlers } from "./hooks/useVideoHandlers";

function App() {
  const [mode, setMode] = useState<AppMode>("live");

  // WebSocket connection
  const {
    connectionStatus,
    speakers,
    sendMessage,
    sendBinaryData,
    clearSpeakers,
  } = useWebSocket("ws://localhost:8080/ws");

  // Mode değiştiğinde speakers'ı temizle
  useEffect(() => {
    clearSpeakers();
  }, [mode, clearSpeakers]);

  // Video handlers
  const {
    handleVideoLinkReady,
    handleVideoUploadReady,
    handleVideoSelect,
    handleVideoStop,
  } = useVideoHandlers({ sendMessage, sendBinaryData });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <AppHeader connectionStatus={connectionStatus} />

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Mode Selector */}
        <div className="mb-8">
          <ModeSelector selectedMode={mode} onModeChange={setMode} />
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 h-[calc(100vh-280px)]">
          {/* Sol Panel - Kamera/Video/Link */}
          <MainContent
            mode={mode}
            connectionStatus={connectionStatus.status}
            onVideoUploadReady={handleVideoUploadReady}
            onVideoStop={handleVideoStop}
            onVideoSelect={handleVideoSelect}
            onVideoLinkReady={handleVideoLinkReady}
          />

          {/* Sağ Panel - Tespit Edilen Kişiler */}
          <div className="lg:col-span-1">
            <DetectedPersonsPanel speakers={speakers} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
