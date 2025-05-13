import os
import logging
import sys
import platform
from datetime import datetime
from typing import Dict, Any

def configure_logging(log_file: str = "emotion_service.log", level: int = logging.INFO):
    """
    Loglama sistemi yapılandırması
    
    Args:
        log_file: Log dosyasının adı
        level: Log seviyesi (logging.DEBUG, logging.INFO, vb.)
        
    Returns:
        logging.Logger: Yapılandırılmış logger nesnesi
    """
    # Log klasörü oluştur
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Logging yapılandırma - dosya adına tarih ekle
    timestamp = datetime.now().strftime("%Y%m%d")
    log_filename = os.path.join(log_dir, f"{timestamp}_{log_file}")
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )
    
    # Alt modüller için logger'ları yapılandır
    logger = logging.getLogger("emotion-service")
    logger.setLevel(logging.INFO)
    
    # Deepface ve OpenCV gibi kütüphanelerin gereksiz loglarını kapat
    logging.getLogger("deepface").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    return logger

def configure_paths():
    """
    Proje için gerekli path'leri konfigüre eder
    """
    # Dinamik proto klasörü ekle
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'proto'))
    
    # Projenin diğer modüllerini de ekle
    module_dir = os.path.dirname(__file__)
    if module_dir not in sys.path:
        sys.path.append(module_dir)
        
def get_server_address():
    """
    Sunucu adresini döndürür
    
    Returns:
        str: host:port formatında sunucu adresi
    """
    port = os.getenv('PORT', '50052')
    host = os.getenv('HOST', '0.0.0.0')
    return f"{host}:{port}"
    
def get_system_info() -> Dict[str, Any]:
    """
    Sistem bilgilerini toplar
    
    Returns:
        Dict[str, Any]: Sistem bilgilerini içeren sözlük
    """
    info = {
        'os': platform.system(),
        'os_version': platform.version(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'hostname': platform.node(),
        'timestamp': datetime.now().isoformat()
    }
    return info
