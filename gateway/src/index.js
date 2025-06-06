const logger = require("./utils/logger");
const dataCache = require("./cache/dataCache");
const websocketServer = require("./server/websocket");

function startApplication() {
  try {
    dataCache.startCleanupTimer();

    const wss = websocketServer.createWebSocketServer();

    setupErrorHandlers();

    return wss;
  } catch (error) {
    logger.error(`Uygulama başlatılamadı: ${error.message}`);
    process.exit(1);
  }
}

function setupErrorHandlers() {
  process.on("uncaughtException", (err) => {
    logger.error(`Yakalanmayan İstisna: ${err.message}`);
    logger.error(err.stack);
  });

  process.on("unhandledRejection", (reason, promise) => {
    logger.error("İşlenmeyen Promise Reddi:", promise);
    logger.error("Sebep:", reason);
  });

  process.on("SIGINT", () => {
    logger.info("Uygulama kapatma isteği alındı");
    dataCache.stopCleanupTimer();
    process.exit(0);
  });
}

// Uygulamayı başlat
const server = startApplication();
module.exports = server; // Test amaçlı dışa aktar
