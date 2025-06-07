import React, { useCallback, useState, useRef } from "react";

interface VideoUploadProps {
  onVideoSelect: (file: File) => void;
}

const VideoUpload: React.FC<VideoUploadProps> = ({ onVideoSelect }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith("video/")) {
          setSelectedFile(file);
          onVideoSelect(file);
        }
      }
    },
    [onVideoSelect]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        const file = files[0];
        setSelectedFile(file);
        onVideoSelect(file);
      }
    },
    [onVideoSelect]
  );

  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <div className="bg-white rounded-2xl overflow-hidden shadow-2xl border border-gray-200">
      <div className="bg-gradient-to-r from-purple-600 to-purple-700 p-4">
        <h3 className="text-white font-semibold flex items-center gap-2">
          📁 Video Yükleme
        </h3>
      </div>

      <div
        className={`aspect-video p-8 border-2 border-dashed transition-all duration-300 cursor-pointer ${
          isDragOver
            ? "border-purple-500 bg-purple-50"
            : selectedFile
            ? "border-green-500 bg-green-50"
            : "border-gray-300 bg-gray-50 hover:border-purple-400 hover:bg-purple-50"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          className="hidden"
        />

        <div className="h-full flex flex-col items-center justify-center text-center">
          {selectedFile ? (
            <>
              <div className="text-green-500 text-6xl mb-4">✅</div>
              <h4 className="text-lg font-semibold text-gray-800 mb-2">
                Video Seçildi
              </h4>
              <p className="text-gray-600 mb-2">{selectedFile.name}</p>
              <p className="text-sm text-gray-500">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </p>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedFile(null);
                }}
                className="mt-4 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
              >
                Videoyu Değiştir
              </button>
            </>
          ) : (
            <>
              <div className="text-gray-400 text-6xl mb-4">
                {isDragOver ? "🎯" : "📤"}
              </div>
              <h4 className="text-lg font-semibold text-gray-800 mb-2">
                {isDragOver ? "Videoyu Buraya Bırakın" : "Video Yükleyin"}
              </h4>
              <p className="text-gray-600 mb-4">
                Videoyu sürükleyip bırakın veya seçmek için tıklayın
              </p>
              <div className="flex flex-wrap gap-2 justify-center text-xs text-gray-500">
                <span className="px-2 py-1 bg-gray-200 rounded">MP4</span>
                <span className="px-2 py-1 bg-gray-200 rounded">AVI</span>
                <span className="px-2 py-1 bg-gray-200 rounded">MOV</span>
                <span className="px-2 py-1 bg-gray-200 rounded">WebM</span>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default VideoUpload;
