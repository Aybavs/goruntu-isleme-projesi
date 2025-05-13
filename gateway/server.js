const WebSocket = require("ws");
const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");
const path = require("path");
const dotenv = require("dotenv");

// .env dosyasını yükle
dotenv.config();

// Log fonksiyonunu yapılandır
const debug = require("debug")(process.env.DEBUG || "gateway:*");
const logger = {
  info: (message) => console.log(`[INFO] ${message}`),
  error: (message) => console.error(`[ERROR] ${message}`),
  warn: (message) => console.warn(`[WARN] ${message}`),
  debug: (message) => debug(message),
};

// Vision Proto tanımını yükle
const VISION_PROTO_PATH = path.join(__dirname, "..", "protos", "vision.proto");
const visionPackageDefinition = protoLoader.loadSync(VISION_PROTO_PATH, {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});
const visionProto = grpc.loadPackageDefinition(visionPackageDefinition).vision;

// Vision servislerinin ayarları
const visionHost = process.env.VISION_SERVICE_HOST || "localhost";
const visionPort = process.env.VISION_SERVICE_PORT || 50051;
const emotionHost = process.env.EMOTION_SERVICE_HOST || "localhost";
const emotionPort = process.env.EMOTION_SERVICE_PORT || 50052;
const speechHost = process.env.SPEECH_SERVICE_HOST || "localhost";
const speechPort = process.env.SPEECH_SERVICE_PORT || 50053;

// Servis istemcilerini oluştur
const visionClient = new visionProto.VisionService(
  `${visionHost}:${visionPort}`,
  grpc.credentials.createInsecure()
);

// Emotion ve Speech Detection servislerine bağlantı
const emotionClient = new visionProto.EmotionService(
  `${emotionHost}:${emotionPort}`,
  grpc.credentials.createInsecure()
);

const speechClient = new visionProto.SpeechDetectionService(
  `${speechHost}:${speechPort}`,
  grpc.credentials.createInsecure()
);

// Veri önbelleğini tutacak nesne
const dataCache = {
  faces: {},           // Yüz verileri önbelleği
  emotions: {},        // Duygu verileri önbelleği
  speechStatus: {},    // Konuşma durumu önbelleği
  lastUpdate: {},      // Son güncelleme zamanları
};

// Kullanılmayan verileri temizleme fonksiyonu
const cleanupInterval = 10000; // 10 saniye
let cleanupTimer = null;

function startCacheCleanupTimer() {
  cleanupTimer = setInterval(() => {
    const now = Date.now();
    for (const faceId in dataCache.lastUpdate) {
      // 10 saniyeden fazla güncellenmeyen verileri temizle
      if (now - dataCache.lastUpdate[faceId] > 10000) {
        delete dataCache.faces[faceId];
        delete dataCache.emotions[faceId];
        delete dataCache.speechStatus[faceId];
        delete dataCache.lastUpdate[faceId];
        logger.debug(`Face ID ${faceId} verileri temizlendi (timeout)`);
      }
    }
  }, cleanupInterval);
}

// WebSocket server kur - port için .env'den değer al
const port = process.env.PORT || 8080;
const host = process.env.HOST || "localhost"; // 0.0.0.0 yerine localhost kullanıyoruz
const wss = new WebSocket.Server({
  port,
  host,
  path: process.env.WS_PATH || "/ws",
  pingInterval: parseInt(process.env.WS_PING_INTERVAL || "30000"),
  // CORS sorunları için origin kontrolü ekledik
  verifyClient: (info) => {
    const origin = info.origin || info.req.headers.origin;
    logger.debug(`WebSocket bağlantı isteği: ${origin}`);
    return true; // Tüm origin'lere izin ver
  },
});

logger.info(
  `WebSocket server running on ws://${host}:${port}${
    process.env.WS_PATH || "/ws"
  }`
);

// Server hata olaylarını yakalayalım
wss.on("error", (error) => {
  logger.error(`WebSocket server error: ${error.message}`);
});

// Önbellek temizleme işlemini başlat
startCacheCleanupTimer();

wss.on("connection", (ws, req) => {
  const clientIP = req.socket.remoteAddress;
  logger.info(`Client connected via WebSocket from ${clientIP}`);

  // Düzenli aralıklarla mevcut veriyi gönder
  const updateInterval = parseInt(process.env.UPDATE_INTERVAL || "100");
  const updateTimer = setInterval(() => {
    sendCombinedData(ws);
  }, updateInterval);

  ws.on("message", async (message) => {
    logger.debug("Received a frame from frontend");

    // Frontend'den gelen frame verisini gRPC'ye gönder
    const frameRequest = { image: message };

    try {
      // Görüntüyü analiz et ve yüz tespit et
      const visionResponse = await new Promise((resolve, reject) => {
        visionClient.AnalyzeFrame(frameRequest, (err, response) => {
          if (err) {
            logger.error("Error calling AnalyzeFrame:", err);
            logger.error(`ERROR DETAILS: ${err.message}`);
            logger.error(`ERROR CODE: ${err.code}`);
            logger.error(`ERROR STACK: ${err.stack}`);
            reject(err);
            return;
          }
          resolve(response);
        });
      });

      // Vision Service'den gelen yanıtları önbelleğe ekle
      processVisionResponse(visionResponse);
      
      // Tespit edilen her yüz için Emotion ve Speech servisleri ile iletişim kur
      if (visionResponse.faces && visionResponse.faces.length > 0) {
        // Tüm yüzler için ayrı ayrı işlem yap
        visionResponse.faces.forEach(face => {
          // Yüz verilerinden gRPC istekleri oluştur
          const faceRequest = {
            face_id: face.id,
            face_image: face.face_image,
            landmarks: face.landmarks
          };
          
          // Duygu analizi için istek gönder
          emotionClient.AnalyzeEmotion(faceRequest, (err, emotionResponse) => {
            if (err) {
              logger.error(`Emotion analizi hatası - Yüz ID ${face.id}: ${err.message}`);
              return;
            }
            // Duygu analizi yanıtını işle
            processEmotionResponse(emotionResponse);
          });
          
          // Konuşma tespiti için istek gönder
          speechClient.DetectSpeech(faceRequest, (err, speechResponse) => {
            if (err) {
              logger.error(`Konuşma tespiti hatası - Yüz ID ${face.id}: ${err.message}`);
              return;
            }
            // Konuşma tespiti yanıtını işle
            processSpeechResponse(speechResponse);
          });
        });
      }
      
      // Anlık bir güncelleme gönder
      sendCombinedData(ws);
    } catch (error) {
      logger.error(`Error processing frame: ${error.message}`);
      logger.error(`ERROR STACK: ${error.stack}`);
      if (error.details) {
        logger.error(`ERROR DETAILS: ${error.details}`);
      }
      if (error.code) {
        logger.error(`ERROR CODE: ${error.code}`);
      }
      ws.send(
        JSON.stringify({ error: `Failed to process frame: ${error.message}` })
      );
    }
  });

  ws.on("close", () => {
    logger.info("Client disconnected");
    // Zamanlayıcıyı temizle
    clearInterval(updateTimer);
  });
});

// Vision Service'den gelen veriyi işle
function processVisionResponse(visionResponse) {
  if (!visionResponse || !visionResponse.faces) return;
  
  const now = Date.now();
  
  // Her yüz için bilgileri önbelleğe ekle/güncelle
  visionResponse.faces.forEach(face => {
    const faceId = face.id;
    
    // Yüz verisini önbelleğe ekle
    dataCache.faces[faceId] = {
      id: faceId,
      face_image: face.face_image,
      x: face.x,
      y: face.y,
      width: face.width,
      height: face.height
    };
    
    // Son güncelleme zamanını kaydet
    dataCache.lastUpdate[faceId] = now;
  });
}

// Emotion Service'den gelen yanıtları işleyecek fonksiyon
function processEmotionResponse(emotionResponse) {
  if (!emotionResponse) return;
  
  const faceId = emotionResponse.face_id;
  const now = Date.now();
  
  // Duygu bilgilerini önbelleğe ekle
  dataCache.emotions[faceId] = {
    emotion: emotionResponse.emotion,
    confidence: emotionResponse.confidence
  };
  
  // Son güncelleme zamanını kaydet
  dataCache.lastUpdate[faceId] = now;
  
  logger.debug(`Duygu analizi güncellendi - Yüz ID: ${faceId}, Duygu: ${emotionResponse.emotion}`);
}

// Speech Service'den gelen yanıtları işleyecek fonksiyon
function processSpeechResponse(speechResponse) {
  if (!speechResponse) return;
  
  const faceId = speechResponse.face_id;
  const now = Date.now();
  
  // Konuşma durumu bilgilerini önbelleğe ekle
  dataCache.speechStatus[faceId] = {
    is_speaking: speechResponse.is_speaking,
    speaking_time: speechResponse.speaking_time
  };
  
  // Son güncelleme zamanını kaydet
  dataCache.lastUpdate[faceId] = now;
  
  logger.debug(`Konuşma durumu güncellendi - Yüz ID: ${faceId}, Konuşuyor: ${speechResponse.is_speaking}`);
}

// Birleştirilmiş veriyi istemciye gönder
function sendCombinedData(ws) {
  // Aktif yüz kimlikleri
  const activeFaceIds = Object.keys(dataCache.faces);
  
  if (activeFaceIds.length === 0) {
    return; // Veri yoksa gönderme
  }
  
  // Birleştirilmiş veriyi hazırla
  const combinedResponse = {
    speakers: activeFaceIds.map(faceId => {
      // Yüz verisi
      const faceData = dataCache.faces[faceId] || {};
      
      // Duygu verisi
      const emotionData = dataCache.emotions[faceId] || { 
        emotion: "unknown", 
        confidence: 0.0 
      };
      
      // Konuşma verisi
      const speechData = dataCache.speechStatus[faceId] || { 
        is_speaking: false, 
        speaking_time: 0.0 
      };
      
      // face_image'i Base64'e dönüştür (eğer varsa)
      let base64FaceImage = null;
      if (faceData.face_image) {
        // Buffer'a dönüştürülmüş değilse dönüştür
        const buffer = Buffer.isBuffer(faceData.face_image) 
          ? faceData.face_image 
          : Buffer.from(faceData.face_image);
        
        base64FaceImage = buffer.toString('base64');
      }
      
      // Birleştirilmiş konuşmacı verisi
      return {
        id: parseInt(faceId),
        face_image: base64FaceImage,
        emotion: emotionData.emotion,
        emotion_confidence: emotionData.confidence,
        is_speaking: speechData.is_speaking,
        speaking_time: speechData.speaking_time
      };
    })
  };
  
  // WebSocket üzerinden veriyi gönder
  try {
    ws.send(JSON.stringify(combinedResponse));
  } catch (err) {
    logger.error(`WebSocket veri gönderme hatası: ${err.message}`);
  }
}

// Yeni kanal abonelikleri için dinleyicileri başlat
function setupEmotionServiceListener() {
  // Speech ve Emotion servisleriyle doğrudan iletişim kurmuyoruz,
  // çünkü Vision Service onlara veri gönderiyor. Bu işleve ihtiyaç yok şu an için.
  logger.info("Gateway üç servis arasındaki iletişime aracılık ediyor");
}

// Log bilgisi
logger.info(`Vision service connects to ${visionHost}:${visionPort}`);
logger.info(`Emotion service runs at ${emotionHost}:${emotionPort}`);
logger.info(`Speech Detection service runs at ${speechHost}:${speechPort}`);

// Hata yakalama
process.on("uncaughtException", (err) => {
  logger.error(`Uncaught Exception: ${err.message}`);
  logger.error(err.stack);
});

process.on("unhandledRejection", (reason, promise) => {
  logger.error("Unhandled Rejection at:", promise);
  logger.error("Reason:", reason);
});
