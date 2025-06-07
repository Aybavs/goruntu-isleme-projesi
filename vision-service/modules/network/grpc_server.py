"""
gRPC sunucu yönetim modülü
"""

import grpc
from concurrent import futures
import proto.vision_pb2_grpc as vision_pb2_grpc
from ..config.logger_config import setup_logger
from ..config.grpc_config import GrpcConfig
from ..core.vision_service import VisionServiceServicer

# Logger'ı yapılandır
logger = setup_logger()


class GrpcServer:
    """gRPC sunucusunu yöneten sınıf"""
    
    def __init__(self):
        """GrpcServer'ı başlatır"""
        self.config = GrpcConfig()
        self.server = None
        
    def create_server(self):
        """gRPC sunucusunu oluşturur"""
        try:
            # gRPC sunucusunu oluştur
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
                options=self.config.grpc_options
            )
            
            # Vision service'i sunucuya ekle
            vision_pb2_grpc.add_VisionServiceServicer_to_server(
                VisionServiceServicer(), 
                self.server
            )
            
            # Port'u ekle
            self.server.add_insecure_port(self.config.address)
            
            logger.info(f"gRPC sunucu oluşturuldu: {self.config.address}")
            return True
            
        except Exception as e:
            logger.error(f"gRPC sunucu oluşturma hatası: {str(e)}")
            return False
    
    def start_server(self):
        """gRPC sunucusunu başlatır"""
        try:
            if not self.server:
                if not self.create_server():
                    return False
            
            self.server.start()
            logger.info(f"VisionService gRPC server is running on {self.config.address}...")
            return True
            
        except Exception as e:
            logger.error(f"gRPC sunucu başlatma hatası: {str(e)}")
            return False
    
    def stop_server(self, grace_period=5):
        """gRPC sunucusunu durdurur"""
        try:
            if self.server:
                self.server.stop(grace_period)
                logger.info("gRPC sunucu durduruldu")
                return True
            return True
            
        except Exception as e:
            logger.error(f"gRPC sunucu durdurma hatası: {str(e)}")
            return False
    
    def wait_for_termination(self):
        """Sunucunun sonlanmasını bekler"""
        try:
            if self.server:
                self.server.wait_for_termination()
                
        except KeyboardInterrupt:
            logger.info("Klavye kesintisi algılandı, sunucu kapatılıyor...")
            self.stop_server()
        except Exception as e:
            logger.error(f"Sunucu bekleme hatası: {str(e)}")
    
    def serve(self):
        """Sunucuyu başlatır ve çalışmasını bekler"""
        if self.start_server():
            try:
                self.wait_for_termination()
            except Exception as e:
                logger.error(f"Sunucu servis hatası: {str(e)}")
            finally:
                self.stop_server()
