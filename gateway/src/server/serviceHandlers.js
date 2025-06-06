const clients = require("../services/clients");
const dataCache = require("../cache/dataCache");
const logger = require("../utils/logger");

// Throttling - İstek sınırlandırma
class RequestThrottler {
  constructor(maxRequestsPerSecond) {
    this.maxRequestsPerSecond = maxRequestsPerSecond;
    this.requestCount = 0;
    this.lastResetTime = Date.now();
  }

  async throttle() {
    const now = Date.now();

    // 1 saniye sonra sayacı sıfırla
    if (now - this.lastResetTime > 1000) {
      this.requestCount = 0;
      this.lastResetTime = now;
    }

    // Limit aşıldıysa, bir sonraki saniyeye kadar bekle
    if (this.requestCount >= this.maxRequestsPerSecond) {
      const waitTime = 1000 - (now - this.lastResetTime);
      if (waitTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, waitTime));
        return this.throttle(); // Bekledikten sonra tekrar kontrol et
      }
    }

    this.requestCount++;
    return true;
  }
}

// Throttler instance oluştur (saniyede maksimum istek sayıları)
const visionThrottler = new RequestThrottler(10);
const emotionThrottler = new RequestThrottler(15);
const speechThrottler = new RequestThrottler(15);

// Gelen bir frame'i işle
async function processFrame(frameData) {
  logger.debug("Frontend'den bir frame alındı");

  // gRPC isteğini oluştur
  const frameRequest = { image: frameData };

  try {
    // Vision servisini retry ve throttling ile çağır
    await visionThrottler.throttle();
    const visionResponse = await callVisionServiceWithRetry(frameRequest);
    dataCache.updateVisionData(visionResponse.faces);

    // Tespit edilen yüzler varsa batch işleme ile işle
    if (visionResponse.faces && visionResponse.faces.length > 0) {
      await batchProcessFaces(visionResponse.faces);
    }

    return visionResponse; // İşlenen frame sonuçlarını döndür
  } catch (error) {
    logger.error("Frame işleme hatası:", error);
    throw error; // Hatanın üst katmanda ele alınması için yeniden fırlat
  }
}

// Vision servisini çağırarak frame'i analiz et
function callVisionService(frameRequest) {
  return new Promise((resolve, reject) => {
    clients.visionClient.AnalyzeFrame(frameRequest, (err, response) => {
      if (err) {
        logGrpcError("AnalyzeFrame", err);
        reject(err);
        return;
      }
      resolve(response);
    });
  });
}

// Tespit edilen bir yüzü işle - throttling eklenmiş
async function processDetectedFace(face) {
  // Duygu ve konuşma servisleri için yüz isteği oluştur
  const faceRequest = {
    face_id: face.id,
    face_image: face.face_image,
    landmarks: face.landmarks,
  };

  try {
    // Emotion servisini throttling ve retry ile çağır
    await emotionThrottler.throttle();
    const emotionResponse = await callWithRetry(
      (req, cb) => clients.emotionClient.AnalyzeEmotion(req, cb),
      faceRequest
    );
    dataCache.updateEmotionData(emotionResponse);
  } catch (err) {
    logger.error(`Duygu analizi hatası - Yüz ID ${face.id}: ${err.message}`);
  }

  try {
    // Speech servisini throttling ve retry ile çağır
    await speechThrottler.throttle();
    const speechResponse = await callWithRetry(
      (req, cb) => clients.speechClient.DetectSpeech(req, cb),
      faceRequest
    );
    dataCache.updateSpeechData(speechResponse);
  } catch (err) {
    logger.error(`Konuşma tespiti hatası - Yüz ID ${face.id}: ${err.message}`);
  }
}

// gRPC hatalarını detaylı şekilde logla
function logGrpcError(serviceName, err) {
  logger.error(`${serviceName} çağrısında hata:`, err);
  logger.error(`HATA DETAYI: ${err.message}`);
  logger.error(`HATA KODU: ${err.code}`);
  logger.error(`HATA STACK: ${err.stack}`);
}

// Retry fonksiyonunu serviceHandlers.js'e ekleyin
async function callWithRetry(serviceFn, request, maxRetries = 3, delay = 1000) {
  let lastError;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await new Promise((resolve, reject) => {
        serviceFn(request, (err, response) => {
          if (err) reject(err);
          else resolve(response);
        });
      });
    } catch (error) {
      lastError = error;
      logger.warn(
        `Deneme ${attempt}/${maxRetries} başarısız: ${error.message}`
      );

      if (attempt < maxRetries) {
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError;
}

// Kullanım örneği:
async function callVisionServiceWithRetry(frameRequest) {
  return callWithRetry(
    (req, cb) => clients.visionClient.AnalyzeFrame(req, cb),
    frameRequest
  );
}

// Batch işleme - Yüz işleme için (serviceHandlers.js'e ekleyin)
async function batchProcessFaces(faces) {
  if (!faces || faces.length === 0) return;

  // Yüzleri gruplar halinde işle
  const BATCH_SIZE = 5;
  const faceChunks = [];

  for (let i = 0; i < faces.length; i += BATCH_SIZE) {
    faceChunks.push(faces.slice(i, i + BATCH_SIZE));
  }

  await Promise.all(
    faceChunks.map(async (chunk) => {
      await Promise.all(chunk.map((face) => processDetectedFace(face)));
    })
  );
}

// Daha açıklayıcı durum kontrolü için yardımcı fonksiyon
function validateResponse(response, serviceName) {
  if (!response) {
    logger.warn(`${serviceName} boş yanıt döndürdü`);
    return false;
  }
  return true;
}

// Servis sağlığını kontrol etmek için kullanılabilecek fonksiyon
async function checkServicesHealth() {
  const healthStatus = {
    vision: false,
    emotion: false,
    speech: false,
  };

  try {
    // Vision service health check
    await callWithRetry(
      (req, cb) => clients.visionClient.HealthCheck({}, cb),
      {},
      1 // Sadece bir deneme
    );
    healthStatus.vision = true;
  } catch (err) {
    logger.error("Vision servisi sağlık kontrolü başarısız:", err.message);
  }

  // Diğer servisler de benzer şekilde kontrol edilebilir

  return healthStatus;
}

module.exports = {
  processFrame,
  callWithRetry,
  batchProcessFaces,
  checkServicesHealth,
  RequestThrottler,
};
