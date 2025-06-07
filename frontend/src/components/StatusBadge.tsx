import React from "react";
import { ConnectionStatus } from "../types";

interface StatusBadgeProps {
  status: ConnectionStatus;
  message: string;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ status, message }) => {
  const getStatusConfig = () => {
    switch (status) {
      case "connecting":
        return {
          bgColor: "bg-gradient-to-r from-yellow-400 to-yellow-500",
          textColor: "text-white",
          icon: "🔄",
          pulse: true,
        };
      case "connected":
        return {
          bgColor: "bg-gradient-to-r from-green-400 to-green-500",
          textColor: "text-white",
          icon: "✅",
          pulse: false,
        };
      case "disconnected":
        return {
          bgColor: "bg-gradient-to-r from-gray-400 to-gray-500",
          textColor: "text-white",
          icon: "⚫",
          pulse: false,
        };
      case "error":
        return {
          bgColor: "bg-gradient-to-r from-red-400 to-red-500",
          textColor: "text-white",
          icon: "❌",
          pulse: false,
        };
      default:
        return {
          bgColor: "bg-gradient-to-r from-gray-400 to-gray-500",
          textColor: "text-white",
          icon: "❓",
          pulse: false,
        };
    }
  };

  const config = getStatusConfig();

  return (
    <div
      className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold shadow-lg ${
        config.bgColor
      } ${config.textColor} ${config.pulse ? "animate-pulse" : ""}`}
    >
      <span className="mr-2 text-base">{config.icon}</span>
      {message}
    </div>
  );
};

export default StatusBadge;
