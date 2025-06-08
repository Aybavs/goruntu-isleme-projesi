"""
Speech Detection Service - gRPC servisi için modül
"""
import time
import logging
import os
from .speech_detector import SpeechDetector
from .speaking_time_tracker import SpeakingTimeTracker

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
        
        # Speaking time tracker nesnesini oluştur
        self.time_tracker = SpeakingTimeTracker(session_timeout=300.0)  # 5 dakika timeout
        
    
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
            
            # Konuşma süresini güncelle ve al
            total_speaking_time = self.time_tracker.update_speaking_status(face_id, is_speaking)
            
            # İstatistikleri al
            stats = self.speech_detector.get_stats(face_id)
            time_stats = self.time_tracker.get_speaking_stats(face_id)
            
            logger.info(f"Yüz ID {face_id} için konuşma durumu: {is_speaking}, Toplam süre: {total_speaking_time:.2f}s")
            
            # Yanıt oluştur
            return vision_pb2.SpeechResponse(
                is_speaking=is_speaking,
                speaking_time=total_speaking_time,
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
        self.time_tracker.clear_face_data(face_id)
        
    def cleanup_inactive_data(self):
        """
        İnaktif yüz verilerini temizler
        """
        cleaned_count = self.time_tracker.cleanup_inactive_faces()
        return cleaned_count
