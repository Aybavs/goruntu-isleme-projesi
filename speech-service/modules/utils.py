"""
Utility functions for speech detection service
"""
import logging
import sys
import os
from datetime import datetime

def setup_logging(log_level='INFO', log_to_file=True, log_file='speech_service.log'):
    """
    Loglama yapılandırmasını ayarlar
    
    Args:
        log_level (str): Loglama seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file (bool): Dosyaya log yazılıp yazılmayacağı
        log_file (str): Log dosyası adı
    """
    # Log seviyesini ayarla
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Handlers listesi oluştur
    handlers = [logging.StreamHandler()]
    
    # Dosyaya yazma aktifse handler ekle
    if log_to_file:
        date_str = datetime.now().strftime('%Y%m%d')
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        
        # Log dizini yoksa oluştur
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Günlük log dosyası
        log_file_path = os.path.join(log_dir, f"{date_str}_{log_file}")
        handlers.append(logging.FileHandler(log_file_path))
        
        # Genel log dosyası
        general_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), log_file)
        handlers.append(logging.FileHandler(general_log_path))
    
    # Loglama yapılandırması
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Bazı ek loggerların seviyelerini ayarla
    logging.getLogger('grpc').setLevel(logging.WARNING)
    
    return logging.getLogger("speech-service")

def add_proto_path():
    """
    Proto dosyaları dizinini Python yoluna ekler
    """
    proto_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'proto')
    if proto_dir not in sys.path:
        sys.path.append(proto_dir)
        
def check_dependencies():
    """
    Bağımlılıkları kontrol eder ve eksikse hata verir
    
    Returns:
        bool: Tüm bağımlılıklar yüklü ise True, değilse False
    """
    dependencies = ["numpy", "grpc"]
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"Eksik bağımlılıklar: {', '.join(missing)}")
        print("Lütfen şu komutu çalıştırın: pip install " + " ".join(missing))
        return False
        
    return True
