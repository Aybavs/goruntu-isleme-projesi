"""
Vision Service gRPC servis implementasyonu
"""

import proto.vision_pb2 as vision_pb2
import proto.vision_pb2_grpc as vision_pb2_grpc
from ..config.logger_config import setup_logger
from ..config.config_manager import ConfigManager
from ..config.grpc_config import GrpcConfig
from ..vision.face_detector import FaceDetector
from ..vision.face_tracker import FaceTracker
from ..vision.frame_processor import FrameProcessor
from ..network.service_client import ServiceClient
from ..network.response_builder import ResponseBuilder

# Logger'ı yapılandır
logger = setup_logger()


class VisionServiceServicer(vision_pb2_grpc.VisionServiceServicer):
    """Vision Service gRPC servis implementasyonu"""
    
    def __init__(self):
        """Vision Service sınıfını başlatır"""
        
        # Yapılandırma yöneticisini başlat
        self.app_config = ConfigManager()
        
        # gRPC yapılandırmasını yükle
        self.config = GrpcConfig()
        
        # Alt modülleri başlat
        self.face_detector = FaceDetector(
            self.app_config.cascade_path,
            self.app_config.model_path
        )
        self.face_tracker = FaceTracker(
            similarity_threshold=self.app_config.face_match_threshold,
            cleanup_timeout=self.app_config.face_cleanup_timeout
        )
        
        # Frame işleyiciyi başlat
        self.frame_processor = FrameProcessor(self.face_detector, self.face_tracker)
        
        # Servis client'ını başlat
        self.service_client = ServiceClient(self.config)
        
        logger.info("Vision Service başlatıldı")

    def AnalyzeFrame(self, request, context):
        """
        Bir görüntü karesini analiz eder ve tespit edilen yüzleri döner
        """
        try:
            # Frame'i işle
            processed_faces, success = self.frame_processor.process_frame(request.image)
            
            if not success:
                logger.error("Frame işleme başarısız.")
                return vision_pb2.VisionResponse(person_detected=False)
            
            # ResponseBuilder kullanarak yanıt oluştur
            response = ResponseBuilder.create_vision_response(processed_faces)
            
            # Her tespit edilen yüzü diğer servislere gönder
            for detected_face in response.faces:
                self._process_detected_face(detected_face)
            
            return response
            
        except Exception as e:
            logger.error(f"AnalyzeFrame hatası: {str(e)}")
            return vision_pb2.VisionResponse(person_detected=False)
        
    def _process_detected_face(self, detected_face):
        """Tespit edilen yüzü Emotion Service ve Speech Detection Service'e gönderir"""
        
        # ResponseBuilder kullanarak FaceRequest oluştur
        face_request = ResponseBuilder.create_face_request(detected_face)
        
        if face_request:
            # ServiceClient kullanarak asenkron olarak gönder
            self.service_client.process_detected_face_async(face_request)
