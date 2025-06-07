
import cv2
import numpy as np
import time
from .logger_config import setup_logger

logger = setup_logger()


class FrameProcessor:
    """Görüntü karelerini işlemek için kullanılan sınıf"""
    
    def __init__(self, face_detector, face_tracker):
        """
        Frame işleyicisini başlatır
        
        Args:
            face_detector: Yüz tespit modülü
            face_tracker: Yüz takip modülü
        """
        self.face_detector = face_detector
        self.face_tracker = face_tracker
        logger.info("Frame işleyici başlatıldı")
    
    def decode_frame(self, image_data):
        """
        Frame verisini numpy array'e çevirir
        
        Args:
            image_data: Binary görüntü verisi
            
        Returns:
            tuple: (img, success) - Decode edilmiş görüntü ve başarı durumu
        """
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Gelen görüntü decode edilemedi.")
                return None, False
                
            return img, True
        except Exception as e:
            logger.error(f"Frame decode hatası: {str(e)}")
            return None, False
    
    def extract_face_region(self, img, face_coords):
        """
        Yüz bölgesini görüntüden keser
        
        Args:
            img: Ana görüntü
            face_coords: (x, y, w, h) yüz koordinatları
            
        Returns:
            numpy.ndarray: Kesilmiş yüz görüntüsü
        """
        x, y, w, h = face_coords
        return img[y:y+h, x:x+w].copy()
    
    def encode_face_image(self, face_img):
        """
        Yüz görüntüsünü encode eder
        
        Args:
            face_img: Yüz görüntüsü
            
        Returns:
            tuple: (encoded_data, success) - Encode edilmiş veri ve başarı durumu
        """
        try:
            is_success, encoded_img = cv2.imencode('.jpg', face_img)
            if is_success:
                return bytes(encoded_img), True
            else:
                logger.warning("Yüz görüntüsü encode edilemedi")
                return None, False
        except Exception as e:
            logger.error(f"Yüz görüntüsü encode hatası: {str(e)}")
            return None, False
    
    def process_frame(self, image_data):
        """
        Bir görüntü karesini tam olarak işler
        
        Args:
            image_data: Binary görüntü verisi
            
        Returns:
            tuple: (processed_faces, success) - İşlenmiş yüzler listesi ve başarı durumu
        """
        # Frame'i decode et
        img, decode_success = self.decode_frame(image_data)
        if not decode_success:
            return [], False
        
        current_time = time.time()
        processed_faces = []
        
        try:
            # Yüzleri tespit et
            faces, gray = self.face_detector.detect_faces(img)
            logger.info(f"Frame analizi: {len(faces)} yüz tespit edildi")
            
            # Her yüzü işle
            for (x, y, w, h) in faces:
                try:
                    face_data = self._process_single_face(img, gray, (x, y, w, h), current_time)
                    if face_data:
                        processed_faces.append(face_data)
                except Exception as face_error:
                    logger.error(f"Yüz işleme hatası: {str(face_error)}")
                    continue
            
            # Eski yüzleri temizle
            self.face_tracker.clean_old_faces(current_time)
            
            return processed_faces, True
            
        except Exception as e:
            logger.error(f"Frame işleme hatası: {str(e)}")
            return [], False
    
    def _process_single_face(self, img, gray, face_coords, current_time):
        """
        Tek bir yüzü işler
        
        Args:
            img: Ana görüntü
            gray: Gri tonlamalı görüntü
            face_coords: (x, y, w, h) yüz koordinatları
            current_time: Mevcut zaman
            
        Returns:
            dict: İşlenmiş yüz verisi
        """
        x, y, w, h = face_coords
        
        # Yüz landmark noktalarını al
        landmarks = self.face_detector.get_landmarks(gray, face_coords)
        
        # Yüz özniteliklerini çıkar
        face_encoding = self.face_detector.extract_face_features(img, face_coords)
        
        # Yüzü tanımla
        face_id = self.face_tracker.identify_face(face_encoding, current_time)
        
        # Yüz bölgesini kes
        face_img = self.extract_face_region(img, face_coords)
        
        # Yüz görüntüsünü encode et
        encoded_face, encode_success = self.encode_face_image(face_img)
        
        return {
            'id': face_id,
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'landmarks': landmarks,
            'face_image': encoded_face if encode_success else None
        }
