"""
gRPC servis bağlantılarını yöneten modül
"""

import grpc
import threading
from ..config.logger_config import setup_logger
import proto.vision_pb2_grpc as vision_pb2_grpc

# Logger'ı yapılandır
logger = setup_logger()


class ServiceClient:
    """gRPC servis bağlantılarını yöneten sınıf"""
    
    def __init__(self, config):
        """
        ServiceClient'ı başlatır
        
        Args:
            config: GrpcConfig nesnesi
        """
        self.config = config
        self.emotion_stub = None
        self.speech_stub = None
        
        # Servis bağlantılarını oluştur
        self._create_service_stubs()
    
    def _create_service_stubs(self):
        """Diğer servislere bağlantı için stub'ları oluşturur"""
        try:
            # Emotion Service'e bağlantı
            emotion_channel = grpc.insecure_channel(
                self.config.emotion_service_address,
                options=self.config.grpc_channel_options
            )
            self.emotion_stub = vision_pb2_grpc.EmotionServiceStub(emotion_channel)
            logger.info(f"Emotion Service'e bağlantı hazırlandı: {self.config.emotion_service_address}")
            
            # Speech Service'e bağlantı
            speech_channel = grpc.insecure_channel(
                self.config.speech_service_address,
                options=self.config.grpc_channel_options
            )
            self.speech_stub = vision_pb2_grpc.SpeechDetectionServiceStub(speech_channel)
            logger.info(f"Speech Detection Service'e bağlantı hazırlandı: {self.config.speech_service_address}")
            
        except Exception as e:
            logger.error(f"Servis bağlantıları oluşturulurken hata: {str(e)}")
            # Hata durumunda stub'ları None olarak ayarla
            self.emotion_stub = None
            self.speech_stub = None
    
    def send_to_emotion_service(self, face_request):
        """Emotion Service'e istek gönderir"""
        try:
            if self.emotion_stub:
                logger.info(f"Emotion Service'e istek gönderiliyor (Yüz ID: {face_request.face_id})")
                response = self.emotion_stub.AnalyzeEmotion(face_request)
                logger.info(f"Emotion Service'den yanıt alındı: {response.emotion} ({response.confidence:.2f})")
                return response
        except Exception as e:
            logger.error(f"Emotion Service isteği başarısız: {str(e)}")
            return None
    
    def send_to_speech_service(self, face_request):
        """Speech Detection Service'e istek gönderir"""
        try:
            if self.speech_stub:
                logger.info(f"Speech Detection Service'e istek gönderiliyor (Yüz ID: {face_request.face_id})")
                response = self.speech_stub.DetectSpeech(face_request)
                logger.info(f"Speech Service'den yanıt alındı: Konuşuyor: {response.is_speaking}, Süre: {response.speaking_time:.2f}s")
                return response
        except Exception as e:
            logger.error(f"Speech Service isteği başarısız: {str(e)}")
            return None
    
    def process_detected_face_async(self, face_request):
        """Tespit edilen yüzü asenkron olarak diğer servislere gönderir"""
        
        # Duygu analizi için isteği gönder
        try:
            if self.emotion_stub:
                threading.Thread(
                    target=self.send_to_emotion_service,
                    args=(face_request,)
                ).start()
        except Exception as e:
            logger.error(f"Emotion Service'e gönderme hatası: {str(e)}")
        
        # Konuşma tespiti için isteği gönder
        try:
            if self.speech_stub:
                threading.Thread(
                    target=self.send_to_speech_service,
                    args=(face_request,)
                ).start()
        except Exception as e:
            logger.error(f"Speech Service'e gönderme hatası: {str(e)}")
