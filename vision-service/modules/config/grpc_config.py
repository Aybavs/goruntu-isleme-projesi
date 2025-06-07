import os
import logging

logger = logging.getLogger("vision-service")

class GrpcConfig:
    """gRPC yapılandırma ayarları"""
    
    def __init__(self):
        """gRPC ayarlarını çevresel değişkenlerden yükler"""
        self.host = os.getenv('HOST', '0.0.0.0')
        self.port = os.getenv('PORT', '50051')
        self.max_workers = int(os.getenv('MAX_WORKERS', '10'))
        
        # Mesaj boyut limitleri
        self.max_message_size = 50 * 1024 * 1024  # 50MB
        
        # Diğer servis adresleri
        self.emotion_service_address = os.getenv('EMOTION_SERVICE_ADDRESS', 'localhost:50052')
        self.speech_service_address = os.getenv('SPEECH_SERVICE_ADDRESS', 'localhost:50053')
        
        logger.info(f"gRPC yapılandırması yüklendi: {self.host}:{self.port}")
    
    @property
    def address(self):
        """Tam adres string'i döner"""
        return f"{self.host}:{self.port}"
    
    @property
    def grpc_options(self):
        """gRPC server seçeneklerini döner"""
        return [
            ('grpc.max_send_message_length', self.max_message_size),
            ('grpc.max_receive_message_length', self.max_message_size)
        ]
    
    @property
    def grpc_channel_options(self):
        """gRPC kanal seçeneklerini döner"""
        return [
            ('grpc.max_send_message_length', 10 * 1024 * 1024),  # 10MB
            ('grpc.max_receive_message_length', 10 * 1024 * 1024)  # 10MB
        ]
