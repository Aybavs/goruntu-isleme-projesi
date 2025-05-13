"""
Duygu Analizi Servisi
---------------------
Bu modül, yüz görüntülerinden duygu analizi yapan bir gRPC servisi sağlar.

Modüler yapı:
- modules/emotion_analyzer.py: Duygu analizi için ana sınıf
- modules/service.py: gRPC servis implementasyonu
- modules/utils.py: Yardımcı fonksiyonlar

Kullanım:
    python emotion_server.py [--port PORT] [--host HOST] [--workers WORKERS] [--confidence CONFIDENCE]
    
Örnek:
    python emotion_server.py --port 50052 --host 0.0.0.0 --workers 10 --confidence 0.35
"""
import grpc
from concurrent import futures
import os
import signal
import sys
import argparse
import time
import threading
from datetime import datetime

# Modüllerimizi import et
from modules.utils import configure_logging, configure_paths, get_server_address, get_system_info
from modules.service import EmotionServiceServicer
import proto.vision_pb2_grpc as vision_pb2_grpc

# Proje path'lerini yapılandır
configure_paths()

# Logger'ı yapılandır
logger = configure_logging()

# Sağlık monitörü için global değişkenler
server = None
start_time = None
request_counter = 0
request_counter_lock = threading.Lock()

def signal_handler(sig, frame):
    """Sinyal yakalayıcı - Ctrl+C için graceful shutdown"""
    if server:
        logger.info("Sunucu kapatma sinyali alındı, servis sonlandırılıyor...")
        server.stop(5)  # 5 saniye içinde mevcut istekleri tamamla
        logger.info("Sunucu başarıyla durduruldu.")
    sys.exit(0)

def increment_request_counter():
    """İstek sayacını artırır"""
    global request_counter
    with request_counter_lock:
        request_counter += 1
        return request_counter

def health_monitor(interval=60):
    """Sunucu sağlık bilgilerini belirli aralıklarla loglar"""
    global start_time, request_counter
    
    while True:
        uptime = time.time() - start_time
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        with request_counter_lock:
            current_count = request_counter
            
        logger.info(f"Sağlık durumu: Çalışma süresi: {int(hours)}s {int(minutes)}d {int(seconds)}s, "
                   f"Toplam istek: {current_count}")
        
        # Sistem bilgilerini log
        system_info = get_system_info()
        logger.debug(f"Sistem bilgileri: {system_info}")
        
        time.sleep(interval)

def parse_arguments():
    """Komut satırı argümanlarını işler"""
    parser = argparse.ArgumentParser(description="Duygu Analizi gRPC Servisi")
    parser.add_argument('--host', default='0.0.0.0', help='Sunucu host adresi')
    parser.add_argument('--port', type=int, default=50052, help='Sunucu port numarası')
    parser.add_argument('--workers', type=int, default=10, help='İş parçacığı sayısı')
    parser.add_argument('--confidence', type=float, default=0.35, help='Duygu tespiti güven eşiği (0.0-1.0)')
    return parser.parse_args()

def serve(args=None):
    """gRPC sunucusunu başlatır"""
    global server, start_time
    
    if args is None:
        args = parse_arguments()
    
    # Environment variables'dan veya argümanlardan adres bilgilerini al
    host = os.getenv('HOST', args.host)
    port = os.getenv('PORT', args.port)
    address = f"{host}:{port}"
    
    # Başlangıç zamanını kaydet
    start_time = time.time()
    
    # Sinyal yakalayıcıları tanımla
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Sunucuyu başlat
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        interceptors=[]  # Burada gerekli interceptor'lar eklenebilir
    )
    
    # Servis nesnesi oluştur ve sunucuya ekle
    service = EmotionServiceServicer(confidence_threshold=args.confidence)
    vision_pb2_grpc.add_EmotionServiceServicer_to_server(service, server)
    
    # Sunucuyu belirtilen adreste başlat
    server.add_insecure_port(address)
    server.start()
    
    # Sistem bilgilerini log'la
    system_info = get_system_info()
    logger.info(f"Sistem bilgileri: {system_info}")
    
    # Sunucu başlangıç bilgilerini log'la
    logger.info(f"EmotionService gRPC servisi başlatıldı:")
    logger.info(f" - Adres: {address}")    
    logger.info(f" - İş parçacığı sayısı: {args.workers}")
    logger.info(f" - Güven eşiği: {args.confidence}")
    logger.info(f" - Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
      # Sağlık monitörünü ayrı bir thread'de başlat
    health_thread = threading.Thread(target=health_monitor, args=(300,), daemon=True)
    health_thread.start()
    
    # Sunucuyu çalışır durumda tut
    try:
        logger.info("Sunucu hazır, istekleri bekliyor...")
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Klavye kesintisi alındı, servis kapatılıyor...")
        server.stop(0)
    except Exception as e:
        logger.error(f"Beklenmeyen bir hata oluştu: {str(e)}")
        server.stop(0)
        sys.exit(1)

if __name__ == "__main__":
    serve()
