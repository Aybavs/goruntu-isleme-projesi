const WebSocket = require("ws");
const config = require("../config");
const logger = require("../utils/logger");
const dataCache = require("../cache/dataCache");
const serviceHandlers = require("./serviceHandlers");

function createWebSocketServer() {
  const wss = new WebSocket.Server({
    port: config.server.port,
    host: config.server.host,
    path: config.server.wsPath,
    pingInterval: config.server.pingInterval,
    verifyClient: (info) => {
      const origin = info.origin || info.req.headers.origin;
      logger.debug(`WebSocket bağlantı isteği: ${origin}`);
      return true; // Tüm origin'lere izin ver
    },
  });

  logger.info(
    `WebSocket sunucusu çalışıyor: ws://${config.server.host}:${config.server.port}${config.server.wsPath}`
  );

  // Sunucu hatalarını yakala
  wss.on("error", (error) => {
    logger.error(`WebSocket sunucu hatası: ${error.message}`);
  });

  // İstemci bağlantılarını yönet
  wss.on("connection", handleClientConnection);

  return wss;
}

function handleClientConnection(ws, req) {
  const clientIP = req.socket.remoteAddress;
  logger.info(`İstemci WebSocket ile bağlandı: ${clientIP}`);

  // Düzenli aralıklarla veri gönder
  const updateTimer = setInterval(() => {
    sendDataToClient(ws);
  }, config.server.updateInterval);

  // Gelen mesajları işle
  ws.on("message", async (message) => {
    try {
      await serviceHandlers.processFrame(message);
      // Anlık bir güncelleme gönder
      sendDataToClient(ws);
    } catch (error) {
      handleError(error, ws);
    }
  });

  // Bağlantı koptuğunda işle
  ws.on("close", () => {
    logger.info("İstemci bağlantısı kesildi");
    clearInterval(updateTimer);
  });
}

function sendDataToClient(ws) {
  try {
    const combinedData = dataCache.getCombinedData();
    if (combinedData.speakers.length > 0) {
      ws.send(JSON.stringify(combinedData));
    }
  } catch (err) {
    logger.error(`WebSocket veri gönderme hatası: ${err.message}`);
  }
}

function handleError(error, ws) {
  logger.error(`Frame işleme hatası: ${error.message}`);
  logger.error(`HATA STACK: ${error.stack}`);

  if (error.details) {
    logger.error(`HATA DETAYI: ${error.details}`);
  }
  if (error.code) {
    logger.error(`HATA KODU: ${error.code}`);
  }

  ws.send(
    JSON.stringify({ error: `Frame işleme başarısız: ${error.message}` })
  );
}

module.exports = {
  createWebSocketServer,
};
