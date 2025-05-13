import React, { JSX } from "react";

import { ConnectionStatus } from "../types";

interface StatusBadgeProps {
  status: ConnectionStatus;
  message: string;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ status, message }) => {
  const statusStyles: Record<
    ConnectionStatus,
    { bg: string; text: string; icon: JSX.Element }
  > = {
    connecting: {
      bg: "bg-yellow-100",
      text: "text-yellow-800",
      icon: (
        <svg
          className="animate-spin -ml-1 mr-2 h-4 w-4 text-yellow-600"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          ></circle>
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          ></path>
        </svg>
      ),
    },
    connected: {
      bg: "bg-green-100",
      text: "text-green-800",
      icon: (
        <svg
          className="h-4 w-4 mr-2 text-green-600"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M5 13l4 4L19 7"
          />
        </svg>
      ),
    },
    disconnected: {
      bg: "bg-gray-100",
      text: "text-gray-800",
      icon: (
        <svg
          className="h-4 w-4 mr-2 text-gray-600"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      ),
    },
    error: {
      bg: "bg-red-100",
      text: "text-red-800",
      icon: (
        <svg
          className="h-4 w-4 mr-2 text-red-600"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
      ),
    },
  };

  const { bg, text, icon } = statusStyles[status];

  return (
    <div
      className={`inline-flex items-center rounded-md ${bg} ${text} px-3 py-1 text-sm font-medium`}
    >
      {icon}
      <span>{message}</span>
    </div>
  );
};

export default StatusBadge;
