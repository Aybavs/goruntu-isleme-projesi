import cv2
import dlib
import numpy as np
import os
import logging

logger = logging.getLogger("vision-service")

class FaceDetector:
    """Yüz tespiti ve landmark analizi için sınıf"""
    
    def __init__(self, cascade_path=None, landmark_path=None):
        """Face detector sınıfını başlatır"""
        # Parametreler verilmediyse varsayılan dosyaları kullan
        if cascade_path is None:
            cascade_path = os.getenv('CASCADE_PATH', 'haarcascade_frontalface_default.xml')
        if landmark_path is None:
            landmark_path = os.getenv('MODEL_PATH', 'shape_predictor_68_face_landmarks.dat')
            
        # Modelleri yükle
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.landmark_predictor = dlib.shape_predictor(landmark_path)
        
        logger.info(f"Yüz tespit modelleri yüklendi: {cascade_path} ve {landmark_path}")
        
    def detect_faces(self, image):
        """
        Görüntüdeki yüzleri algılar
        Args:
            image: OpenCV görüntüsü (BGR)
        Returns:
            [(x, y, w, h)] formatında yüz dikdörtgenlerinin listesi
        """
        # Gri tonlamaya çevir ve gürültü azalt
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Yüzleri tespit et
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2, 
            minNeighbors=5, 
            minSize=(50, 50)
        )
        
        return faces, gray
    
    def get_landmarks(self, gray_image, face_rect):
        """
        Yüz bölgesindeki landmark noktalarını tespit eder
        Args:
            gray_image: Gri tonlamalı görüntü
            face_rect: (x, y, w, h) formatında yüz dikdörtgeni
        Returns:
            Landmark noktalarının [(x,y), (x,y), ...] formatında listesi
        """
        try:
            x, y, w, h = face_rect
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            dlib_landmarks = self.landmark_predictor(gray_image, rect)
            
            # dlib_landmarks nesnesini (x,y) koordinat çiftleri listesine dönüştür
            landmark_points = []
            for i in range(dlib_landmarks.num_parts):
                point = dlib_landmarks.part(i)
                landmark_points.append((point.x, point.y))
                
            return landmark_points
        except Exception as e:
            logger.error(f"Landmark tespiti hatası: {str(e)}")
            return []
        
    def extract_face_features(self, image, face_rect):
        """
        Yüz bölgesinden öznitelikleri çıkarır - daha güçlü bir öznitelik seti kullanır
        Args:
            image: Renkli görüntü
            face_rect: (x, y, w, h) formatında yüz dikdörtgeni
        Returns:
            Yüz öznitelik vektörü (numpy array)
        """
        x, y, w, h = face_rect
        
        # Yüz bölgesini kes
        face_region = image[y:y+h, x:x+w]
        
        # Görüntü çok küçükse işleme
        if face_region.size == 0 or face_region.shape[0] < 10 or face_region.shape[1] < 10:
            return np.zeros(512)  # Boş öznitelik vektörü döndür

        # Yüz bölgesini yeniden boyutlandır
        face_region = cv2.resize(face_region, (100, 100))
        
        # Histogram hesapla (RGB ve HSV uzayında daha gürbüz öznitelik)
        hist_rgb = cv2.calcHist([face_region], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_rgb = cv2.normalize(hist_rgb, hist_rgb).flatten()
        
        # HSV renk uzayında da hesapla
        face_hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        hist_hsv = cv2.calcHist([face_hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        
        # İki histogramı birleştir
        combined_features = np.concatenate((hist_rgb, hist_hsv))
        
        return combined_features
