import cv2
import numpy as np
import sys
import os
import time
import dotenv
from pathlib import Path
import threading
import logging
import grpc

# .env dosyasını yükle
dotenv.load_dotenv(Path(__file__).parent / '.env')

# Dinamik proto klasörü ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'proto'))

# Özel modülleri içe aktar
from modules.face_detector import FaceDetector
from modules.face_tracker import FaceTracker

import proto.vision_pb2 as vision_pb2
import proto.vision_pb2_grpc as vision_pb2_grpc
from concurrent import futures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vision_service.log")
    ]
)
logger = logging.getLogger("vision-service")

class VisionServiceServicer(vision_pb2_grpc.VisionServiceServicer):
    def __init__(self):
        """Vision Service sınıfını başlatır"""
        
        # Çeşitli eşik değerlerini çevresel değişkenlerden al
        face_match_threshold = float(os.getenv('FACE_MATCH_THRESHOLD', '0.7'))
        face_cleanup_timeout = float(os.getenv('FACE_CLEANUP_TIMEOUT', '5.0'))
        
        # Alt modülleri başlat
        self.face_detector = FaceDetector(
            os.getenv('CASCADE_PATH', 'haarcascade_frontalface_default.xml'),
            os.getenv('MODEL_PATH', 'shape_predictor_68_face_landmarks.dat')
        )
        
        self.face_tracker = FaceTracker(
            similarity_threshold=face_match_threshold,
            cleanup_timeout=face_cleanup_timeout
        )
        
        # Duygu analizi ve konuşma tespiti servisleriyle iletişim kurma
        self.emotion_service_address = os.getenv('EMOTION_SERVICE_ADDRESS', 'localhost:50052')
        self.speech_service_address = os.getenv('SPEECH_SERVICE_ADDRESS', 'localhost:50053')
        
        # Servis stub'larını oluştur
        self._create_service_stubs()
        
        logger.info("Vision Service başlatıldı")
    
    def _create_service_stubs(self):
        """Diğer servislere bağlantı için stub'ları oluşturur"""
        try:
            # Emotion Service'e bağlantı
            emotion_channel = grpc.insecure_channel(
                self.emotion_service_address,
                options=[
                    ('grpc.max_send_message_length', 10 * 1024 * 1024),  # 10MB
                    ('grpc.max_receive_message_length', 10 * 1024 * 1024)  # 10MB
                ]
            )
            self.emotion_stub = vision_pb2_grpc.EmotionServiceStub(emotion_channel)
            logger.info(f"Emotion Service'e bağlantı hazırlandı: {self.emotion_service_address}")
            
            # Speech Service'e bağlantı
            speech_channel = grpc.insecure_channel(
                self.speech_service_address,
                options=[
                    ('grpc.max_send_message_length', 10 * 1024 * 1024),  # 10MB
                    ('grpc.max_receive_message_length', 10 * 1024 * 1024)  # 10MB
                ]
            )
            self.speech_stub = vision_pb2_grpc.SpeechDetectionServiceStub(speech_channel)
            logger.info(f"Speech Detection Service'e bağlantı hazırlandı: {self.speech_service_address}")
            
        except Exception as e:
            logger.error(f"Servis bağlantıları oluşturulurken hata: {str(e)}")
            # Hata durumunda stub'ları None olarak ayarla
            self.emotion_stub = None
            self.speech_stub = None

    def AnalyzeFrame(self, request, context):
        """
        Bir görüntü karesini analiz eder ve tespit edilen yüzleri döner
        """
        try:
            # Frame verisini numpy array'e çevir
            nparr = np.frombuffer(request.image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("Gelen görüntü decode edilemedi.")
                # DÜZELTME: VisionResponse yerine vision_pb2 modülündeki gerçek sınıf adını kullan
                return vision_pb2.VisionResponse(person_detected=False)

            current_time = time.time()
            
            # Yüzleri tespit et
            faces, gray = self.face_detector.detect_faces(img)
            
            # DÜZELTME: VisionResponse yerine vision_pb2 modülündeki gerçek sınıf adını kullan
            response = vision_pb2.VisionResponse(
                person_detected=len(faces) > 0
            )
            
            # Her yüzü işle
            for (x, y, w, h) in faces:
                try:
                    # Yüz landmark noktalarını al
                    landmarks = self.face_detector.get_landmarks(gray, (x, y, w, h))
                    
                    # Yüz özniteliklerini çıkar
                    face_encoding = self.face_detector.extract_face_features(img, (x, y, w, h))
                    
                    # Yüzü tanımla
                    face_id = self.face_tracker.identify_face(face_encoding, current_time)
                    
                    # Yüz bölgesini kes
                    face_img = img[y:y+h, x:x+w].copy()
                    
                    # DetectedFace nesnesini oluştur
                    detected_face = vision_pb2.DetectedFace()
                    detected_face.id = face_id
                    
                    # Yüz landmark'larını ekle
                    if landmarks is not None:
                        # Artık landmarks [(x,y), (x,y), ...] formatında bir liste
                        detected_face.landmarks.extend([float(coord) for point in landmarks for coord in point])
                    
                    # Yüz koordinatlarını ekle
                    detected_face.x = x
                    detected_face.y = y
                    detected_face.width = w
                    detected_face.height = h
                    
                    # Yüz görüntüsünü çevirip ekle
                    is_success, encoded_img = cv2.imencode('.jpg', face_img)
                    if is_success:
                        detected_face.face_image = bytes(encoded_img)
                    
                    # Yüzü ana yanıta ekle
                    response.faces.append(detected_face)
                    
                    # İşlenmiş yüz verisini diğer servislere gönder
                    self._process_detected_face(detected_face)
                
                except Exception as face_error:
                    logger.error(f"Yüz işleme hatası: {str(face_error)}")
                    continue
            
            # Eski yüzleri temizle
            self.face_tracker.clean_old_faces(current_time)
            
            logger.info(f"Frame analizi: {len(faces)} yüz tespit edildi")
            return response
            
        except Exception as e:
            logger.error(f"AnalyzeFrame hatası: {str(e)}")
            # DÜZELTME: VisionResponse yerine vision_pb2 modülündeki gerçek sınıf adını kullan
            return vision_pb2.VisionResponse(person_detected=False)
    
    def _process_detected_face(self, detected_face):
        """Tespit edilen yüzü Emotion Service ve Speech Detection Service'e gönderir"""
        
        # FaceRequest nesnesini oluştur
        face_request = vision_pb2.FaceRequest(
            face_image=detected_face.face_image,
            face_id=detected_face.id,
            landmarks=detected_face.landmarks
        )
        
        # Duygu analizi için isteği gönder
        try:
            if self.emotion_stub:
                threading.Thread(
                    target=self._send_to_emotion_service,
                    args=(face_request,)
                ).start()
        except Exception as e:
            logger.error(f"Emotion Service'e gönderme hatası: {str(e)}")
        
        # Konuşma tespiti için isteği gönder
        try:
            if self.speech_stub:
                threading.Thread(
                    target=self._send_to_speech_service,
                    args=(face_request,)
                ).start()
        except Exception as e:
            logger.error(f"Speech Service'e gönderme hatası: {str(e)}")
    
    def _send_to_emotion_service(self, face_request):
        """Emotion Service'e istek gönderir"""
        try:
            logger.info(f"Emotion Service'e istek gönderiliyor (Yüz ID: {face_request.face_id})")
            # PredictEmotion yerine vision.proto'da tanımlanan AnalyzeEmotion kullanıyoruz
            response = self.emotion_stub.AnalyzeEmotion(face_request)
            logger.info(f"Emotion Service'den yanıt alındı: {response.emotion} ({response.confidence:.2f})")
        except Exception as e:
            logger.error(f"Emotion Service isteği başarısız: {str(e)}")
    
    def _send_to_speech_service(self, face_request):
        """Speech Detection Service'e istek gönderir"""
        try:
            logger.info(f"Speech Detection Service'e istek gönderiliyor (Yüz ID: {face_request.face_id})")
            response = self.speech_stub.DetectSpeech(face_request)
            logger.info(f"Speech Service'den yanıt alındı: Konuşuyor: {response.is_speaking}, Süre: {response.speaking_time:.2f}s")
        except Exception as e:
            logger.error(f"Speech Service isteği başarısız: {str(e)}")


def serve():
    """gRPC sunucusunu başlatır"""
    port = os.getenv('PORT', '50051')
    host = os.getenv('HOST', '0.0.0.0')
    address = f"{host}:{port}"
    max_workers = int(os.getenv('MAX_WORKERS', '10'))

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # 50MB
        ]
    )
    vision_pb2_grpc.add_VisionServiceServicer_to_server(VisionServiceServicer(), server)
    server.add_insecure_port(address)
    server.start()
    logger.info(f"VisionService gRPC server is running on {address}...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
