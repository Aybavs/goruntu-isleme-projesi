const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");
const path = require("path");
const config = require("../config");
const logger = require("../utils/logger");

// Proto tanımlarını yükle
function loadProtoDefinitions() {
  const VISION_PROTO_PATH = path.join(
    __dirname,
    "..",
    "..",
    "protos",
    "vision.proto"
  );
  return protoLoader.loadSync(VISION_PROTO_PATH, {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true,
  });
}

// Servis istemcilerini başlat
function initializeClients() {
  const visionPackageDefinition = loadProtoDefinitions();
  const visionProto = grpc.loadPackageDefinition(
    visionPackageDefinition
  ).vision;

  const visionClient = new visionProto.VisionService(
    `${config.vision.host}:${config.vision.port}`,
    grpc.credentials.createInsecure()
  );

  const emotionClient = new visionProto.EmotionService(
    `${config.emotion.host}:${config.emotion.port}`,
    grpc.credentials.createInsecure()
  );

  const speechClient = new visionProto.SpeechDetectionService(
    `${config.speech.host}:${config.speech.port}`,
    grpc.credentials.createInsecure()
  );

  logger.info(
    `Vision servisi bağlantısı: ${config.vision.host}:${config.vision.port}`
  );
  logger.info(
    `Emotion servisi bağlantısı: ${config.emotion.host}:${config.emotion.port}`
  );
  logger.info(
    `Speech Detection servisi bağlantısı: ${config.speech.host}:${config.speech.port}`
  );

  return {
    visionClient,
    emotionClient,
    speechClient,
  };
}

module.exports = initializeClients();
