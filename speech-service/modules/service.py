"""
Speech Detection Service - gRPC servisi için modül
"""
import time
import logging
import os
from .speech_detector import SpeechDetector

# Proto dosyalarını import et
import proto.vision_pb2 as vision_pb2
import proto.vision_pb2_grpc as vision_pb2_grpc

logger = logging.getLogger("speech-service")

class SpeechDetectionServicer(vision_pb2_grpc.SpeechDetectionServiceServicer):
    """Speech Detection Service - gRPC servis sınıfı"""
    
    def __init__(self, config=None):
        """
        Speech Detection Service sınıfını başlatır
        
        Args:
            config (dict, optional): Konfigürasyon ayarları
        """
        # Varsayılan konfigürasyon değerleri
        self.config = {
            'variation_threshold': float(os.getenv('VARIATION_THRESHOLD', '0.03')),
            'confidence_threshold': float(os.getenv('SPEAKING_CONFIDENCE_THRESHOLD', '0.35')),
            'cooldown_frames': int(os.getenv('COOLDOWN_FRAMES', '3')),
            'history_length': int(os.getenv('HISTORY_LENGTH', '20')),
            'adaptation_rate': float(os.getenv('ADAPTATION_RATE', '0.08'))
        }
        
        # Eğer konfigürasyon parametresi geçildiyse değerleri güncelle
        if config:
            self.config.update(config)
        
        # Speech detector nesnesini oluştur
        self.speech_detector = SpeechDetector(
            variation_threshold=self.config['variation_threshold'],
            confidence_threshold=self.config['confidence_threshold'],
            cooldown_frames=self.config['cooldown_frames'],
            history_length=self.config['history_length'],
            adaptation_rate=self.config['adaptation_rate']
        )
        
        logger.info(f"Speech Detection Service başlatıldı: {self.config}")
    
    def DetectSpeech(self, request, context):
        """
        Vision Service'den gelen yüz verisini kullanarak konuşma tespiti yapar
        
        Args:
            request (SpeechRequest): gRPC isteği
            context: gRPC bağlam nesnesi
            
        Returns:
            SpeechResponse: Konuşma durumu yanıtı
        """
        try:
            face_id = request.face_id
            landmarks = list(request.landmarks)
            
            # Ayrıntı seviyesini ayarlamak için giriş doğrulama
            if not face_id or not landmarks:
                logger.warning("Geçersiz istek: face_id veya landmarks eksik")
                return vision_pb2.SpeechResponse(
                    is_speaking=False,
                    speaking_time=0.0,
                    face_id=request.face_id
                )
            
            logger.debug(f"Konuşma tespiti isteği alındı (Yüz ID: {face_id})")
            
            # Mevcut zaman
            current_time = time.time()
            
            # Konuşma durumunu tespit et
            is_speaking = self.speech_detector.detect_speaking(face_id, landmarks, current_time)
            
            # Konuşma süresini al
            speaking_time = self.speech_detector.get_speaking_time(face_id)
            
            # İstatistikleri al (isteğe bağlı olarak yanıta eklenebilir)
            stats = self.speech_detector.get_face_stats(face_id)
            
            logger.info(f"Yüz ID {face_id} için konuşma durumu: {is_speaking}, süre: {speaking_time:.2f} sn")
            
            # Yanıt oluştur
            return vision_pb2.SpeechResponse(
                is_speaking=is_speaking,
                speaking_time=speaking_time,
                face_id=face_id
            )
            
        except Exception as e:
            logger.error(f"Konuşma tespiti hatası: {str(e)}", exc_info=True)
            return vision_pb2.SpeechResponse(
                is_speaking=False,
                speaking_time=0.0,
                face_id=request.face_id
            )
            
    def clear_face_data(self, face_id):
        """
        Yüz verilerini temizler (servis arayüzünden doğrudan erişim için)
        
        Args:
            face_id (str): Temizlenecek yüz kimliği
        """
        self.speech_detector.clear_face_data(face_id)
