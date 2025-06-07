// websocket.js
const WebSocket = require("ws");
const config = require("../config");
const logger = require("../utils/logger");
const dataCache = require("../cache/dataCache");
const serviceHandlers = require("./serviceHandlers");
const ytdl = require("@distube/ytdl-core");
const ffmpeg = require("fluent-ffmpeg");
const ffmpegInstaller = require("@ffmpeg-installer/ffmpeg");
ffmpeg.setFfmpegPath(ffmpegInstaller.path);

function createWebSocketServer() {
  const wss = new WebSocket.Server({
    port: config.server.port,
    path: config.server.wsPath,
    pingInterval: config.server.pingInterval,
    verifyClient: (info) => {
      const origin = info.origin || info.req.headers.origin;
      logger.debug(`WebSocket bağlantı isteği: ${origin}`);
      return true;
    },
  });

  logger.info(
    `WebSocket sunucusu çalışıyor: ws://<host>:${config.server.port}${config.server.wsPath}`
  );

  wss.on("error", (error) => {
    logger.error(`WebSocket sunucu hatası: ${error.message}`);
  });

  wss.on("connection", handleClientConnection);
  // Uygulama kapatılırken temizlik
  process.on("SIGINT", () => {
    logger.info("WebSocket sunucusu kapatılıyor...");
    wss.close();
  });

  return wss;
}

function handleClientConnection(ws, req) {
  const clientIP = req.socket.remoteAddress;
  logger.info(`İstemci WebSocket ile bağlandı: ${clientIP}`);

  let clientData = {
    activeYouTubeStreams: new Map(),
    updateTimer: null,
  };

  // Düzenli veri güncelleme
  clientData.updateTimer = setInterval(() => {
    sendDataToClient(ws);
  }, config.server.updateInterval);

  ws.on("message", async (data, isBinary) => {
    // 1) Binary data ise doğrudan frame işle
    if (isBinary) {
      try {
        await serviceHandlers.processFrame(data);
        sendDataToClient(ws);
      } catch (err) {
        handleError(err, ws);
      }
      return;
    }

    // 2) Text ise JSON komut işleme
    let parsed;
    try {
      parsed = JSON.parse(data.toString());
    } catch {
      return ws.send(JSON.stringify({ error: "Geçersiz mesaj formatı" }));
    }

    try {
      // YouTube analizi komutu
      if (parsed.type === "analyzeYoutube" && parsed.youtubeUrl) {
        await handleYouTubeAnalysis(parsed, ws, clientData);
      }
      // YouTube analizi durdurma komutu
      else if (parsed.type === "stopYoutubeAnalysis" && parsed.streamId) {
        handleStopYouTubeAnalysis(parsed, ws, clientData);
      }
      // Aktif stream sayısını getir
      else if (parsed.type === "getActiveStreams") {
        ws.send(
          JSON.stringify({
            event: "activeStreamsCount",
            count: clientData.activeYouTubeStreams.size,
            clientStreams: Array.from(clientData.activeYouTubeStreams.keys()),
          })
        );
      } else {
        ws.send(JSON.stringify({ error: "Bilinmeyen komut tipi" }));
      }
    } catch (error) {
      logger.error(`Komut işleme hatası: ${error.message}`);
      ws.send(
        JSON.stringify({ error: `Komut işleme hatası: ${error.message}` })
      );
    }
  });

  ws.on("close", () => {
    logger.info("İstemci bağlantısı kesildi");
    cleanup(clientData);
  });

  ws.on("error", (error) => {
    logger.error(`WebSocket client hatası: ${error.message}`);
    cleanup(clientData);
  });
}

async function handleYouTubeAnalysis(parsed, ws, clientData) {
  const { youtubeUrl } = parsed;

  try {
    // YouTube URL'sini doğrula
    if (!ytdl.validateURL(youtubeUrl)) {
      ws.send(JSON.stringify({ error: "Geçersiz YouTube linki" }));
      return;
    }

    // Analiz başladığını bildir
    ws.send(
      JSON.stringify({
        event: "analysisStarted",
        youtubeUrl,
      })
    );

    logger.info(`YouTube analizi başlatılıyor: ${youtubeUrl}`);

    // Video bilgisini al
    let info;
    try {
      info = await ytdl.getInfo(youtubeUrl);
      ws.send(
        JSON.stringify({
          event: "videoInfo",
          youtubeUrl,
          info: {
            title: info.videoDetails.title,
            duration: info.videoDetails.lengthSeconds,
            thumbnail: info.videoDetails.thumbnails[0]?.url,
          },
        })
      );
    } catch (infoError) {
      logger.error(`Video bilgisi alınamadı: ${infoError.message}`);
      ws.send(
        JSON.stringify({
          error: `Video bilgisi alınamadı: ${infoError.message}`,
        })
      );
      return;
    }

    // Stream ID oluştur
    const streamId = `yt_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    try {
      // YouTube video stream'ini başlat
      const videoStream = ytdl(youtubeUrl, {
        quality: "highestvideo",
        filter: "videoonly",
      });

      // FFmpeg ile frame'leri ayıkla
      const ffmpegProcess = ffmpeg(videoStream)
        .format("image2pipe")
        .outputOptions([
          "-vf",
          "fps=1", // 1 frame per second
          "-q:v",
          "2", // High quality JPEG
        ])
        .on("start", (commandLine) => {
          logger.info(`FFmpeg başlatıldı: ${streamId}`);
          logger.debug(`FFmpeg komutu: ${commandLine}`);
          clientData.activeYouTubeStreams.set(streamId, ffmpegProcess);
          ws.send(
            JSON.stringify({
              event: "streamStarted",
              youtubeUrl,
              streamId,
            })
          );
        })
        .on("progress", (progress) => {
          logger.debug(`FFmpeg progress: ${JSON.stringify(progress)}`);
        })
        .on("end", () => {
          logger.info(`YouTube analizi tamamlandı: ${streamId}`);
          clientData.activeYouTubeStreams.delete(streamId);
          ws.send(
            JSON.stringify({
              event: "analysisComplete",
              streamId,
              youtubeUrl,
            })
          );
        })
        .on("error", (err) => {
          logger.error(`FFmpeg hatası: ${err.message}`);
          clientData.activeYouTubeStreams.delete(streamId);
          ws.send(
            JSON.stringify({
              error: `Video işleme hatası: ${err.message}`,
              streamId,
            })
          );
        })
        .pipe();

      // Frame'leri işle
      let buffer = Buffer.alloc(0);
      let frameCount = 0;

      ffmpegProcess.on("data", (chunk) => {
        buffer = Buffer.concat([buffer, chunk]);

        // JPEG başlangıç ve bitiş marker'larını ara
        let start = buffer.indexOf(Buffer.from([0xff, 0xd8])); // JPEG start
        let end = buffer.indexOf(Buffer.from([0xff, 0xd9]), start + 2); // JPEG end

        while (start !== -1 && end !== -1) {
          const frameBuffer = buffer.slice(start, end + 2);
          buffer = buffer.slice(end + 2);
          frameCount++;

          logger.debug(
            `Frame ${frameCount} işleniyor, boyut: ${frameBuffer.length} bytes`
          );

          // Frame'i işle
          (async () => {
            try {
              await serviceHandlers.processFrame(frameBuffer);
              sendDataToClient(ws);
            } catch (err) {
              logger.error(`Frame ${frameCount} işleme hatası: ${err.message}`);
            }
          })();

          // Sonraki frame'i ara
          start = buffer.indexOf(Buffer.from([0xff, 0xd8]));
          end = buffer.indexOf(Buffer.from([0xff, 0xd9]), start + 2);
        }
      });

      ffmpegProcess.on("error", (err) => {
        logger.error(`Stream error: ${err.message}`);
      });
    } catch (streamError) {
      logger.error(`Stream oluşturma hatası: ${streamError.message}`);
      ws.send(
        JSON.stringify({
          error: `Video stream hatası: ${streamError.message}`,
        })
      );
    }
  } catch (error) {
    logger.error("YouTube analizi genel hatası:", error);
    ws.send(
      JSON.stringify({
        error: `YouTube analizi başarısız: ${error.message}`,
      })
    );
  }
}

function handleStopYouTubeAnalysis(parsed, ws, clientData) {
  const { streamId } = parsed;

  try {
    const ffmpegProcess = clientData.activeYouTubeStreams.get(streamId);
    if (ffmpegProcess) {
      ffmpegProcess.kill("SIGTERM");
      clientData.activeYouTubeStreams.delete(streamId);
      ws.send(
        JSON.stringify({
          event: "streamStopped",
          streamId,
        })
      );
      logger.info(`YouTube stream durduruldu: ${streamId}`);
    } else {
      ws.send(
        JSON.stringify({
          error: "Stream bulunamadı veya durdurulamadı",
        })
      );
    }
  } catch (error) {
    logger.error(`Stream durdurma hatası: ${error.message}`);
    ws.send(
      JSON.stringify({
        error: `Stream durdurma hatası: ${error.message}`,
      })
    );
  }
}

function cleanup(clientData) {
  // Timer'ı temizle
  if (clientData.updateTimer) {
    clearInterval(clientData.updateTimer);
    clientData.updateTimer = null;
  }

  // Aktif YouTube streamleri durdur
  for (const [streamId, ffmpegProcess] of clientData.activeYouTubeStreams) {
    try {
      ffmpegProcess.kill("SIGTERM");
      logger.info(`Client kapanırken stream durduruldu: ${streamId}`);
    } catch (err) {
      logger.error(`Stream durdurulamadı: ${err.message}`);
    }
  }
  clientData.activeYouTubeStreams.clear();
}

function sendDataToClient(ws) {
  try {
    if (ws.readyState === WebSocket.OPEN) {
      const combinedData = dataCache.getCombinedData();
      if (combinedData.speakers.length > 0) {
        ws.send(
          JSON.stringify({
            type: "analysisData",
            data: combinedData,
            timestamp: Date.now(),
          })
        );
      }
    }
  } catch (err) {
    logger.error(`WebSocket veri gönderme hatası: ${err.message}`);
  }
}

function handleError(error, ws) {
  logger.error(`Frame işleme hatası: ${error.message}`);
  logger.error(error.stack);
  if (error.details) logger.error(`HATA DETAYI: ${error.details}`);
  if (error.code) logger.error(`HATA KODU: ${error.code}`);

  try {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "error",
          error: `Frame işleme başarısız: ${error.message}`,
          timestamp: Date.now(),
        })
      );
    }
  } catch (sendError) {
    logger.error(`Hata mesajı gönderilemedi: ${sendError.message}`);
  }
}

module.exports = {
  createWebSocketServer,
};
