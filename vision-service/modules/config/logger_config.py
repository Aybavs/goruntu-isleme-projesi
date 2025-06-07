"""
Vision Service için logging yapılandırması
"""
import logging
import os
from pathlib import Path


def setup_logger(name="vision-service", log_file="vision_service.log", level=logging.INFO):
    """
    Logger'ı yapılandırır ve döner
    
    Args:
        name (str): Logger adı
        log_file (str): Log dosyası adı
        level: Log seviyesi
    
    Returns:
        logging.Logger: Yapılandırılmış logger
    """
    # logs klasörünün tam yolunu oluştur
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)  # logs klasörünü oluştur (yoksa)
    
    # Log dosyasının tam yolunu oluştur
    log_path = logs_dir / log_file
    
    # Logging yapılandırması
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path)
        ]
    )
    
    return logging.getLogger(name)


def get_logger(name="vision-service"):
    """
    Mevcut logger'ı döner
    
    Args:
        name (str): Logger adı
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
