import React from "react";
import StatusBadge from "./StatusBadge";
import { ConnectionStatus } from "../types";

interface AppHeaderProps {
  connectionStatus: {
    status: ConnectionStatus;
    message: string;
  };
}

const AppHeader: React.FC<AppHeaderProps> = ({ connectionStatus }) => {
  return (
    <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-10">
      <div className="max-w-7xl mx-auto px-6 py-4 relative">
        {/* Merkezi başlık */}
        <div className="text-center">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Görüntü İşleme Analiz Sistemi
          </h1>
          <p className="text-gray-600 mt-1">
            AI destekli yüz tanıma ve duygu analizi
          </p>
        </div>

        {/* Sağ üst köşede küçük status badge */}
        <div className="absolute top-2 right-6 scale-75">
          <StatusBadge
            status={connectionStatus.status}
            message={connectionStatus.message}
          />
        </div>
      </div>
    </header>
  );
};

export default AppHeader;
