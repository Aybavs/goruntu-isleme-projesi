const dotenv = require("dotenv");
dotenv.config();

module.exports = {
  server: {
    port: process.env.PORT || 8080,
    host: process.env.HOST || "localhost",
    wsPath: process.env.WS_PATH || "/ws",
    pingInterval: parseInt(process.env.WS_PING_INTERVAL || "30000"),
    updateInterval: parseInt(process.env.UPDATE_INTERVAL || "100"),
  },
  vision: {
    host: process.env.VISION_SERVICE_HOST || "localhost",
    port: process.env.VISION_SERVICE_PORT || 50051,
  },
  emotion: {
    host: process.env.EMOTION_SERVICE_HOST || "localhost",
    port: process.env.EMOTION_SERVICE_PORT || 50052,
  },
  speech: {
    host: process.env.SPEECH_SERVICE_HOST || "localhost",
    port: process.env.SPEECH_SERVICE_PORT || 50053,
  },
  cache: {
    cleanupInterval: 10000, // 10 saniye
    dataTimeout: 10000, // 10 saniye
  },
  debug: process.env.DEBUG || "gateway:*",
};
