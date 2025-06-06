const logger = require("../utils/logger");
const config = require("../config");

class DataCache {
  constructor() {
    this.data = {
      faces: {},
      emotions: {},
      speechStatus: {},
      lastUpdate: {},
    };
    this.cleanupTimer = null;
  }

  // Önbellek temizleme işlemini başlat
  startCleanupTimer() {
    const cleanupInterval = config.cache.cleanupInterval || 10000;
    this.cleanupTimer = setInterval(() => this.cleanup(), cleanupInterval);
    logger.debug(
      `Önbellek temizleme zamanlayıcısı başlatıldı (aralık: ${cleanupInterval}ms)`
    );
  }

  // Eski verileri temizle
  cleanup() {
    const now = Date.now();
    const timeout = config.cache.dataTimeout || 10000;

    for (const faceId in this.data.lastUpdate) {
      if (now - this.data.lastUpdate[faceId] > timeout) {
        this.removeFace(faceId);
      }
    }
  }

  // Bir yüzü ve ilişkili verileri kaldır
  removeFace(faceId) {
    delete this.data.faces[faceId];
    delete this.data.emotions[faceId];
    delete this.data.speechStatus[faceId];
    delete this.data.lastUpdate[faceId];
    logger.debug(`Yüz ID ${faceId} önbellekten temizlendi (timeout)`);
  }

  // Vision verilerini güncelle
  updateVisionData(faces) {
    if (!faces) return;

    const now = Date.now();
    faces.forEach((face) => {
      const faceId = face.id;

      this.data.faces[faceId] = {
        id: faceId,
        face_image: face.face_image,
        x: face.x,
        y: face.y,
        width: face.width,
        height: face.height,
      };

      this.data.lastUpdate[faceId] = now;
    });
  }

  // Duygu verilerini güncelle
  updateEmotionData(emotionResponse) {
    if (!emotionResponse) return;

    const faceId = emotionResponse.face_id;
    const now = Date.now();

    this.data.emotions[faceId] = {
      emotion: emotionResponse.emotion,
      confidence: emotionResponse.confidence,
    };

    this.data.lastUpdate[faceId] = now;
    logger.debug(
      `Duygu analizi güncellendi - Yüz ID: ${faceId}, Duygu: ${emotionResponse.emotion}`
    );
  }

  // Konuşma verilerini güncelle
  updateSpeechData(speechResponse) {
    if (!speechResponse) return;

    const faceId = speechResponse.face_id;
    const now = Date.now();

    this.data.speechStatus[faceId] = {
      is_speaking: speechResponse.is_speaking,
      speaking_time: speechResponse.speaking_time,
    };

    this.data.lastUpdate[faceId] = now;
    logger.debug(
      `Konuşma durumu güncellendi - Yüz ID: ${faceId}, Konuşuyor: ${speechResponse.is_speaking}`
    );
  }

  // Tüm aktif yüzler için birleştirilmiş veri al
  getCombinedData() {
    const activeFaceIds = Object.keys(this.data.faces);

    if (activeFaceIds.length === 0) {
      return { speakers: [] };
    }

    return {
      speakers: activeFaceIds.map((faceId) => {
        // Yüz verisi
        const faceData = this.data.faces[faceId] || {};

        // Duygu verisi
        const emotionData = this.data.emotions[faceId] || {
          emotion: "unknown",
          confidence: 0.0,
        };

        // Konuşma verisi
        const speechData = this.data.speechStatus[faceId] || {
          is_speaking: false,
          speaking_time: 0.0,
        };

        // face_image'i Base64'e dönüştür
        let base64FaceImage = null;
        if (faceData.face_image) {
          const buffer = Buffer.isBuffer(faceData.face_image)
            ? faceData.face_image
            : Buffer.from(faceData.face_image);

          base64FaceImage = buffer.toString("base64");
        }

        // Birleştirilmiş konuşmacı verisi
        return {
          id: parseInt(faceId),
          face_image: base64FaceImage,
          emotion: emotionData.emotion,
          emotion_confidence: emotionData.confidence,
          is_speaking: speechData.is_speaking,
          speaking_time: speechData.speaking_time,
        };
      }),
    };
  }

  // Temizleme zamanlayıcısını durdur
  stopCleanupTimer() {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
      logger.debug("Önbellek temizleme zamanlayıcısı durduruldu");
    }
  }
}

module.exports = new DataCache();
