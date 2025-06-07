const ytdl = require("ytdl-core");
const ffmpeg = require("fluent-ffmpeg");
const ffmpegInstaller = require("@ffmpeg-installer/ffmpeg");
const config = require("../config");
const logger = require("../utils/logger");

ffmpeg.setFfmpegPath(ffmpegInstaller.path);

class YouTubeStreamManager {
  constructor() {
    this.activeStreams = new Map();
    this.maxConcurrentStreams = config.youtube.maxConcurrentStreams;
  }

  async startYouTubeAnalysis(youtubeUrl, ws, frameProcessor) {
    // URL doğrulama
    if (!ytdl.validateURL(youtubeUrl)) {
      throw new Error("Geçersiz YouTube linki");
    }

    // Eşzamanlı stream limiti kontrolü
    if (this.activeStreams.size >= this.maxConcurrentStreams) {
      throw new Error(
        `Maksimum ${this.maxConcurrentStreams} eşzamanlı stream destekleniyor`
      );
    }

    const streamId = this.generateStreamId(youtubeUrl);

    // Zaten işlenmekte olan stream kontrolü
    if (this.activeStreams.has(streamId)) {
      throw new Error("Bu video zaten işlenmekte");
    }

    logger.info(`YouTube analizi başlatılıyor: ${youtubeUrl}`);

    try {
      const streamData = await this.createVideoStream(
        youtubeUrl,
        ws,
        frameProcessor
      );
      this.activeStreams.set(streamId, streamData);

      // Stream tamamlandığında temizle
      streamData.cleanup = () => {
        this.activeStreams.delete(streamId);
        logger.info(`YouTube stream temizlendi: ${streamId}`);
      };

      return streamData;
    } catch (error) {
      logger.error(`YouTube stream oluşturma hatası: ${error.message}`);
      throw error;
    }
  }

  async createVideoStream(youtubeUrl, ws, frameProcessor) {
    return new Promise((resolve, reject) => {
      const videoStream = ytdl(youtubeUrl, {
        quality: config.youtube.quality,
        filter: "videoonly",
      });

      const ffmpegCommand = ffmpeg(videoStream)
        .format("image2pipe")
        .fps(config.youtube.frameRate)
        .outputOptions(["-vcodec mjpeg", "-f image2pipe", "-vf scale=640:480"]);

      let buffer = Buffer.alloc(0);
      let isActive = true;
      let frameCount = 0;

      // Timeout ayarla
      const timeout = setTimeout(() => {
        if (isActive) {
          logger.warn(`YouTube stream timeout: ${youtubeUrl}`);
          cleanup();
          reject(new Error("YouTube stream timeout"));
        }
      }, config.youtube.timeout);

      const cleanup = () => {
        isActive = false;
        clearTimeout(timeout);
        try {
          ffmpegCommand.kill("SIGKILL");
        } catch (err) {
          logger.warn(`FFmpeg temizleme hatası: ${err.message}`);
        }
      };

      ffmpegCommand
        .on("start", (commandLine) => {
          logger.info("FFmpeg başladı:", commandLine);
          ws.send(
            JSON.stringify({
              event: "analysisStarted",
              youtubeUrl,
              message: "Video analizi başlatıldı",
            })
          );
        })
        .on("end", () => {
          logger.info("FFmpeg tamamlandı");
          cleanup();
          ws.send(
            JSON.stringify({
              event: "analysisComplete",
              youtubeUrl,
              framesProcessed: frameCount,
              message: "Video analizi tamamlandı",
            })
          );
        })
        .on("error", (err) => {
          logger.error("FFmpeg hatası:", err);
          cleanup();
          reject(err);
        })
        .pipe();

      ffmpegCommand.on("data", async (chunk) => {
        if (!isActive) return;

        try {
          buffer = Buffer.concat([buffer, chunk]);

          // JPEG frame başlangıç ve bitiş markerları
          let start = buffer.indexOf(Buffer.from([0xff, 0xd8]));
          let end = buffer.indexOf(Buffer.from([0xff, 0xd9]), start + 2);

          while (start !== -1 && end !== -1 && isActive) {
            const frameBuf = buffer.slice(start, end + 2);
            buffer = buffer.slice(end + 2);

            // Frame'i işle
            try {
              await frameProcessor(frameBuf);
              frameCount++;

              // İlerleme bilgisi gönder
              if (frameCount % 10 === 0) {
                ws.send(
                  JSON.stringify({
                    event: "analysisProgress",
                    youtubeUrl,
                    framesProcessed: frameCount,
                  })
                );
              }
            } catch (err) {
              logger.error(`Frame işleme hatası: ${err.message}`);
            }

            // Yeni frame arama
            start = buffer.indexOf(Buffer.from([0xff, 0xd8]));
            end = buffer.indexOf(Buffer.from([0xff, 0xd9]), start + 2);
          }
        } catch (err) {
          logger.error(`Chunk işleme hatası: ${err.message}`);
        }
      });

      resolve({
        streamId: this.generateStreamId(youtubeUrl),
        command: ffmpegCommand,
        cleanup,
        isActive: () => isActive,
      });
    });
  }

  stopStream(streamId) {
    const stream = this.activeStreams.get(streamId);
    if (stream && stream.cleanup) {
      stream.cleanup();
      return true;
    }
    return false;
  }

  stopAllStreams() {
    for (const [streamId, stream] of this.activeStreams) {
      if (stream.cleanup) {
        stream.cleanup();
      }
    }
    this.activeStreams.clear();
    logger.info("Tüm YouTube streamleri durduruldu");
  }

  getActiveStreamsCount() {
    return this.activeStreams.size;
  }

  generateStreamId(youtubeUrl) {
    // URL'den video ID'sini çıkar
    const videoId = ytdl.getVideoID(youtubeUrl);
    return `youtube_${videoId}_${Date.now()}`;
  }

  async getVideoInfo(youtubeUrl) {
    try {
      const info = await ytdl.getInfo(youtubeUrl);
      return {
        title: info.videoDetails.title,
        duration: info.videoDetails.lengthSeconds,
        author: info.videoDetails.author.name,
        viewCount: info.videoDetails.viewCount,
      };
    } catch (error) {
      logger.error(`Video bilgisi alınamadı: ${error.message}`);
      return null;
    }
  }
}

module.exports = new YouTubeStreamManager();
