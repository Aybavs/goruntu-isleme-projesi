"""
Uygulama yapılandırma değerlerini yöneten modül
"""

import os
from .logger_config import setup_logger

# Logger'ı yapılandır
logger = setup_logger()


class ConfigManager:
    """Uygulama yapılandırma değerlerini yöneten sınıf"""
    
    def __init__(self):
        """ConfigManager'ı başlatır ve tüm yapılandırma değerlerini yükler"""
        self._load_configuration()
        logger.info("Yapılandırma değerleri yüklendi")
    
    def _load_configuration(self):
        """Environment değişkenlerinden yapılandırma değerlerini yükler"""
        
        # Yüz tespit yapılandırmaları
        self.face_match_threshold = float(os.getenv('FACE_MATCH_THRESHOLD', '0.4'))
        self.face_cleanup_timeout = float(os.getenv('FACE_CLEANUP_TIMEOUT', '5.0'))
        
        # Model dosya yolları
        self.cascade_path = os.getenv('CASCADE_PATH', 'haarcascade_frontalface_default.xml')
        self.model_path = os.getenv('MODEL_PATH', 'shape_predictor_68_face_landmarks.dat')
        
        # Logging yapılandırması
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', 'logs/vision_service.log')
        
        # Debug modları
        self.debug_mode = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
        self.save_debug_images = os.getenv('SAVE_DEBUG_IMAGES', 'False').lower() == 'true'
        
        logger.info(f"Yüz eşleştirme eşiği: {self.face_match_threshold}")
        logger.info(f"Yüz temizleme zaman aşımı: {self.face_cleanup_timeout}s")
        logger.info(f"Debug modu: {self.debug_mode}")
    
    @property
    def face_detector_config(self):
        """FaceDetector için yapılandırma döner"""
        return {
            'cascade_path': self.cascade_path,
            'model_path': self.model_path
        }
    
    @property
    def face_tracker_config(self):
        """FaceTracker için yapılandırma döner"""
        return {
            'similarity_threshold': self.face_match_threshold,
            'cleanup_timeout': self.face_cleanup_timeout
        }
    
    def get_config_summary(self):
        """Yapılandırma özetini döner"""
        return {
            'face_match_threshold': self.face_match_threshold,
            'face_cleanup_timeout': self.face_cleanup_timeout,
            'cascade_path': self.cascade_path,
            'model_path': self.model_path,
            'debug_mode': self.debug_mode,
            'log_level': self.log_level
        }
