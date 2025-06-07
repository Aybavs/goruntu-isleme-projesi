"""
Vision Service Ana Başlatıcı
Bu dosya Vision Service'i başlatmak için kullanılır.
"""

import sys
import os
import dotenv
from pathlib import Path

# .env dosyasını yükle
dotenv.load_dotenv(Path(__file__).parent / '.env')

# Dinamik proto klasörü ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'proto'))

# Özel modülleri içe aktar
from modules import setup_logger, GrpcServer

# Logger'ı yapılandır
logger = setup_logger()


def main():
    """Ana başlatıcı fonksiyon"""
    try:
        logger.info("Vision Service başlatılıyor...")
        server = GrpcServer()
        server.serve()
    except KeyboardInterrupt:
        logger.info("Vision Service kapatılıyor...")
    except Exception as e:
        logger.error(f"Vision Service başlatma hatası: {str(e)}")
        raise


if __name__ == "__main__":
    main()
