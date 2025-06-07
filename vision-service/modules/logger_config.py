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
    # Log dosyasının tam yolunu oluştur
    log_path = Path(__file__).parent.parent / log_file
    
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
