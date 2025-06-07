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
        """        # Varsayılan konfigürasyon değerleri
        
        
        # Speech detector nesnesini oluştur
        self.speech_detector = SpeechDetector()
        
    
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
            
            logger.info(f"Konuşma tespiti isteği alındı (Yüz ID: {face_id}, Landmark sayısı: {len(landmarks)})")
            
            
            # Ayrıntı seviyesini ayarlamak için giriş doğrulama
            if not face_id or not landmarks:
                logger.warning("Geçersiz istek: face_id veya landmarks eksik")
                return vision_pb2.SpeechResponse(
                    is_speaking=False,
                    speaking_time=0.0,
                    face_id=request.face_id
                )
            logger.debug(f"Konuşma tespiti isteği alındı (Yüz ID: {face_id})")
            
            # Konuşma durumunu tespit et
            is_speaking = self.speech_detector.detect_speaking(face_id, landmarks)
            
            # İstatistikleri al
            stats = self.speech_detector.get_stats(face_id)
            
            logger.info(f"Yüz ID {face_id} için konuşma durumu: {is_speaking}")
            
            # Yanıt oluştur
            return vision_pb2.SpeechResponse(
                is_speaking=is_speaking,
                speaking_time=0.0,  # Yeni API'de speaking_time yok, 0.0 gönderiyoruz
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
        self.speech_detector.clear_face(face_id)
