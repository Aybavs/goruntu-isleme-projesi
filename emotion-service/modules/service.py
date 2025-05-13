"""
Duygu Analizi Servisi - gRPC Servis Implementasyonu
-------------------------------------------------
Bu modül, gRPC servis implementasyonunu içerir ve yüz görüntülerinden
duygu analizi yapmak için EmotionAnalyzer sınıfını kullanır.
"""
import cv2
import numpy as np
import logging
import time
import grpc
import traceback
from typing import Dict, Any, Optional

from .emotion_analyzer import EmotionAnalyzer

# Proto içe aktarma
import proto.vision_pb2 as vision_pb2
import proto.vision_pb2_grpc as vision_pb2_grpc

# Eğer emotion_server modülü varsa istek sayacını içe aktar
try:
    from emotion_server import increment_request_counter
    has_request_counter = True
except ImportError:
    has_request_counter = False
    def increment_request_counter():
        return 0

# Logging
logger = logging.getLogger("emotion-service")

class EmotionServiceServicer(vision_pb2_grpc.EmotionServiceServicer):
    """
    Duygu analizi gRPC servisi. 
    
    Bu servis, yüz görüntülerini alıp duygu analizi yapar ve sonuçları döndürür.
    EmotionAnalyzer sınıfını kullanarak görüntüleri işler ve duygu durumlarını tespit eder.
    """
    
    def __init__(self, confidence_threshold: float = 0.35):
        """
        Emotion Service sınıfını başlatır
        
        Args:
            confidence_threshold: Duygu tespitinde kullanılacak minimum güven eşiği (0.0-1.0 arası)
        """
        self.emotion_analyzer = EmotionAnalyzer(confidence_threshold=confidence_threshold)
        logger.info("Geliştirilmiş Emotion Service başlatıldı (Güven eşiği: %.2f)", confidence_threshold)
        
    def AnalyzeEmotion(self, request, context):
        """
        Vision Service'den gelen yüz görüntüsünü analiz edip duygu durumunu belirler
        
        Args:
            request: gRPC istek nesnesi (FaceRequest)
            context: gRPC bağlam nesnesi
            
        Returns:
            vision_pb2.EmotionResponse: Tespit edilen duygu durumu, güven skoru ve yüz ID bilgisi
        """
        # İstek sayacını artır
        if has_request_counter:
            request_count = increment_request_counter()
            
        start_time = time.time()
        logger.info(f"Duygu analizi isteği alındı (Yüz ID: {request.face_id})")
        
        try:
            # Giriş doğrulama
            if not request.face_image:
                logger.error("Boş yüz görüntüsü alındı")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Yüz görüntüsü boş olamaz")
                return vision_pb2.EmotionResponse(emotion="error", confidence=0.0, face_id=request.face_id)
            
            # Yüz görüntüsünü numpy array'e çevir
            nparr = np.frombuffer(request.face_image, np.uint8)
            face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if face_img is None or face_img.size == 0:
                logger.error("Geçersiz yüz görüntüsü formatı")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Geçersiz görüntü formatı")
                return vision_pb2.EmotionResponse(emotion="error", confidence=0.0, face_id=request.face_id)

            # Geliştirilmiş duygu analizi
            logger.debug("Duygu analizi için yüz ön işleme yapılıyor...")
            
            try:
                # Yüz ön işleme
                processed_face = self.emotion_analyzer.preprocess_face(face_img)
                
                # Duygu analizi
                result = self.emotion_analyzer.analyze_emotion(
                    processed_face, 
                    face_id=request.face_id
                )
                
                dominant_emotion = result["emotion"]
                emotion_confidence = result["confidence"]
                
                # İşlem süresini hesapla
                process_time = time.time() - start_time
                
                logger.info(f"Yüz ID {request.face_id} için duygu analizi: {dominant_emotion}, güven: {emotion_confidence:.2f}, süre: {process_time:.3f}s")

                # Sonucu döndür
                return vision_pb2.EmotionResponse(
                    emotion=dominant_emotion,
                    confidence=float(emotion_confidence),
                    face_id=request.face_id
                )
            except Exception as inner_e:
                logger.error(f"Duygu analizi işlem hatası: {str(inner_e)}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Duygu analizi işlem hatası: {str(inner_e)}")
                return vision_pb2.EmotionResponse(
                    emotion="error", 
                    confidence=0.0,
                    face_id=request.face_id
                )
                
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Yüz analiz işleme hatası: {str(e)}\n{error_trace}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Genel işlem hatası: {str(e)}")
            return vision_pb2.EmotionResponse(
                emotion="error", 
                confidence=0.0,
                face_id=request.face_id
            )
        finally:
            # İstek sayısını güncelle
            if has_request_counter:
                logger.info(f"Toplam istek sayısı: {request_count}")
            
            # İşlem süresini logla
            end_time = time.time()
            logger.debug(f"İşlem süresi: {end_time - start_time:.3f} saniye")
            