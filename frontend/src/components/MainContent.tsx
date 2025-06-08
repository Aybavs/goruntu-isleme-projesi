import React from "react";
import LiveCamera from "./LiveCamera";
import VideoUpload from "./VideoUpload";
import VideoLink from "./VideoLink";
import { AppMode, ConnectionStatus } from "../types";

interface MainContentProps {
  mode: AppMode;
  connectionStatus: ConnectionStatus;
  onVideoUploadReady: (videoElement: HTMLVideoElement) => () => void;
  onVideoStop: () => void;
  onVideoSelect: (file: File) => void;
  onVideoLinkReady: (
    videoElement: HTMLVideoElement | null,
    originalUrl: string
  ) => void;
}

const MainContent: React.FC<MainContentProps> = ({
  mode,
  connectionStatus,
  onVideoUploadReady,
  onVideoStop,
  onVideoSelect,
  onVideoLinkReady,
}) => {
  const renderVideoComponent = () => {
    switch (mode) {
      case "live":
        return (
          <LiveCamera
            onVideoReady={onVideoUploadReady}
            onVideoStop={onVideoStop}
            connectionStatus={connectionStatus}
          />
        );
      case "upload":
        return <VideoUpload onVideoSelect={onVideoSelect} />;
      case "link":
        return <VideoLink onVideoLoad={onVideoLinkReady} />;
      default:
        return null;
    }
  };

  return <div className="lg:col-span-2">{renderVideoComponent()}</div>;
};

export default MainContent;
