import cv2
import numpy as np
from deepface import DeepFace
import logging
import threading
import time
import os
import json
from collections import deque
from pathlib import Path
import hashlib

logger = logging.getLogger("vision-service")

class EmotionAnalyzer:
    """Duygu analizi için geliştirilmiş sınıf"""
    
    def __init__(self, confidence_threshold=0.30, config_path=None, enable_caching=True):
        """
        Emotion Analyzer sınıfını başlatır
        Args:
            confidence_threshold: Duygu tespiti için minimum güven skoru
            config_path: Yapılandırma dosyası yolu (JSON formatında)
            enable_caching: DeepFace sonuçlarını önbelleğe almayı etkinleştirir
        """
        self.emotion_confidence_threshold = confidence_threshold
        self.emotion_lock = threading.Lock()  # Thread güvenliği için
        self.enable_caching = enable_caching
        
        # Önbellekleme için değişkenler
        self.cache_size = 50  # Maksimum önbellek boyutu
        self.result_cache = {}  # Yüz hash -> analiz sonucu eşleştirmesi
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # Kullanılacak modeller ve ağırlıkları
        self.models = {
            "deepface": {
                "weight": 1.0,
                "enabled": True,
                "backend": "opencv" # Veya "ssd", "mtcnn", "retinaface"
            }
            # Diğer modeller buraya eklenebilir
            # "custom_model": {"weight": 0.5, "enabled": False, "path": "model.h5"}
        }
        
        # Duygu sınıflandırıcısını kalibre etme değerleri - daha hassas kalibre
        self.emotion_calibration = {
            "angry": {"threshold": 0.40, "boost": 1.25, "decay": 0.85, "priority": 2},
            "disgust": {"threshold": 0.45, "boost": 1.10, "decay": 0.90, "priority": 1},
            "fear": {"threshold": 0.42, "boost": 1.15, "decay": 0.85, "priority": 1},
            "happy": {"threshold": 0.30, "boost": 1.30, "decay": 0.80, "priority": 3},
            "sad": {"threshold": 0.38, "boost": 1.20, "decay": 0.85, "priority": 2},
            "surprise": {"threshold": 0.35, "boost": 1.20, "decay": 0.80, "priority": 2},
            "neutral": {"threshold": 0.25, "boost": 0.95, "decay": 0.70, "priority": 0}
        }
        
        # Duygular arası geçişleri düzenleyen benzerlik matrisi (1: Çok benzer, 0: Hiç benzer değil)
        self.emotion_similarity = {
            "angry": {"disgust": 0.6, "fear": 0.3, "happy": 0.0, "sad": 0.4, "surprise": 0.2, "neutral": 0.1},
            "disgust": {"angry": 0.6, "fear": 0.4, "happy": 0.0, "sad": 0.5, "surprise": 0.1, "neutral": 0.2},
            "fear": {"angry": 0.3, "disgust": 0.4, "happy": 0.0, "sad": 0.5, "surprise": 0.7, "neutral": 0.2},
            "happy": {"angry": 0.0, "disgust": 0.0, "fear": 0.0, "sad": 0.0, "surprise": 0.4, "neutral": 0.3},
            "sad": {"angry": 0.4, "disgust": 0.5, "fear": 0.5, "happy": 0.0, "surprise": 0.1, "neutral": 0.5},
            "surprise": {"angry": 0.2, "disgust": 0.1, "fear": 0.7, "happy": 0.4, "sad": 0.1, "neutral": 0.2},
            "neutral": {"angry": 0.1, "disgust": 0.2, "fear": 0.2, "happy": 0.3, "sad": 0.5, "surprise": 0.2}
        }
        
        # Türkçe duygu isimleri (front-end için)
        self.emotion_tr = {
            "angry": "kızgın",
            "disgust": "iğrenme",
            "fear": "korku",
            "happy": "mutlu",
            "sad": "üzgün",
            "surprise": "şaşkın",
            "neutral": "nötr",
            "uncertain": "belirsiz",
            "unknown": "bilinmiyor",
            "error": "hata"
        }
        
        # Yüz ifade tespitini iyileştirmek için özellik maskesi
        self.face_regions = {
            "angry": [(0.3, 0.7, 0.0, 0.6)],  # Kaşlar ve göz bölgesi
            "disgust": [(0.2, 0.8, 0.4, 0.8)],  # Burun ve ağız
            "fear": [(0.2, 0.8, 0.1, 0.6)],  # Göz ve kaşlar
            "happy": [(0.2, 0.8, 0.5, 0.9)],  # Ağız bölgesi
            "sad": [(0.3, 0.7, 0.3, 0.8)],  # Gözler ve ağız kenarları
            "surprise": [(0.2, 0.8, 0.1, 0.7)]  # Göz, kaşlar ve ağız
        }
        
        # Son duygu geçmişi, stabil tespitler için
        self.emotion_history = {}
        self.emotion_temporal_scores = {}
        self.history_max_size = 7  # Artırıldı - daha uzun tarihçe ile daha iyi analiz
        self.emotion_stability = {}
        
        # Her duygu için duygu skorlarının ortalamasını tut
        self.emotion_averages = {}
        
        # Duygu geçiş soğutma süresi (ani değişimleri engeller)
        self.emotion_cooldown = {}
        self.cooldown_frames = 2
        
        # Kişi bazlı adaptif kalibrasyon değerleri
        self.person_calibration = {}
        self.adaptation_rate = 0.05
        
        # Harici yapılandırma dosyasını yükle (varsa)
        if config_path:
            self._load_config(config_path)
        
        # Günlük bellek temizleme zamanlayıcısı başlat
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 300  # 5 dakikada bir temizlik
        
        logger.info("Geliştirilmiş duygu analizi modülü başlatıldı")
        
    def _load_config(self, config_path):
        """
        Harici yapılandırma dosyasını yükler
        Args:
            config_path: JSON formatındaki yapılandırma dosyasının yolu
        """
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Yapılandırmadan değerleri yükle
                if "threshold" in config:
                    self.emotion_confidence_threshold = config["threshold"]
                
                if "emotion_calibration" in config:
                    self.emotion_calibration.update(config["emotion_calibration"])
                
                if "models" in config:
                    self.models.update(config["models"])
                
                if "cache_size" in config:
                    self.cache_size = config["cache_size"]
                
                if "cleanup_interval" in config:
                    self._cleanup_interval = config["cleanup_interval"]
                    
                logger.info(f"Yapılandırma dosyası başarıyla yüklendi: {config_path}")
            else:
                logger.warning(f"Yapılandırma dosyası bulunamadı: {config_path}")
        except Exception as e:
            logger.error(f"Yapılandırma dosyası yüklenirken hata oluştu: {str(e)}")
    
    def save_config(self, config_path):
        """
        Mevcut yapılandırmayı bir dosyaya kaydeder
        Args:
            config_path: Yapılandırmanın kaydedileceği dosya yolu
        """
        try:
            config = {
                "threshold": self.emotion_confidence_threshold,
                "emotion_calibration": self.emotion_calibration,
                "models": self.models,
                "cache_size": self.cache_size,
                "cleanup_interval": self._cleanup_interval
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            logger.info(f"Yapılandırma dosyası başarıyla kaydedildi: {config_path}")
        except Exception as e:
            logger.error(f"Yapılandırma dosyası kaydedilirken hata oluştu: {str(e)}")
    
    def _compute_image_hash(self, image):
        """
        Görüntü için benzersiz bir hash değeri hesaplar
        Args:
            image: Hash değeri hesaplanacak görüntü
        Returns:
            Görüntünün hash değeri
        """
        if image is None or image.size == 0:
            return None
        
        # Görüntüyü küçült ve hash hesapla
        try:
            small = cv2.resize(image, (32, 32))
            # Basit bir perceptual hash
            small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.normalize(small_gray, None, 0, 255, cv2.NORM_MINMAX)
            img_hash = hashlib.md5(small_gray.tobytes()).hexdigest()
            return img_hash
        except Exception as e:
            logger.error(f"Görüntü hash hesaplama hatası: {str(e)}")
            return None
    
    def _cleanup_old_data(self):
        """
        Belleği optimize etmek için eski verileri temizler
        """
        current_time = time.time()
        # Belirli aralıklarla temizlik yap
        if current_time - self._last_cleanup_time > self._cleanup_interval:
            try:
                # Aktif olmayan yüz verilerini temizle
                inactive_faces = []
                for face_id in self.emotion_history:
                    if face_id not in self.emotion_temporal_scores or len(self.emotion_history[face_id]) == 0:
                        inactive_faces.append(face_id)
                    else:
                        last_record = self.emotion_history[face_id][-1]
                        time_since_last = current_time - last_record.get("timestamp", 0)
                        # 10 dakikadır aktif olmayan yüzleri temizle
                        if time_since_last > 600:  
                            inactive_faces.append(face_id)
                
                # Temizleme işlemi
                for face_id in inactive_faces:
                    if face_id in self.emotion_history:
                        del self.emotion_history[face_id]
                    if face_id in self.emotion_temporal_scores:
                        del self.emotion_temporal_scores[face_id]
                    if face_id in self.emotion_stability:
                        del self.emotion_stability[face_id]
                    if face_id in self.emotion_cooldown:
                        del self.emotion_cooldown[face_id]
                    if face_id in self.person_calibration:
                        del self.person_calibration[face_id]
                
                # Önbelleği temizle (LRU stratejisi)
                if len(self.result_cache) > self.cache_size:
                    # Önbelleğin %20'sini temizle
                    num_to_remove = int(self.cache_size * 0.2)
                    keys_to_remove = list(self.result_cache.keys())[:num_to_remove]
                    for key in keys_to_remove:
                        del self.result_cache[key]
                
                # Önbellek istatistiklerini logla
                if self.enable_caching:
                    total_queries = self.cache_hit_count + self.cache_miss_count
                    hit_rate = (self.cache_hit_count / total_queries * 100) if total_queries > 0 else 0
                    logger.debug(f"Önbellek istatistikleri - Hit: {self.cache_hit_count}, Miss: {self.cache_miss_count}, Oran: {hit_rate:.2f}%")
                
                logger.debug(f"Bellek temizleme tamamlandı - {len(inactive_faces)} yüz verisi temizlendi")
                self._last_cleanup_time = current_time
            except Exception as e:
                logger.error(f"Bellek temizleme hatası: {str(e)}")
    
    def preprocess_face(self, face_img):
        """
        Duygu analizi için yüz görüntüsünü ön işle
        Args:
            face_img: Orijinal yüz görüntüsü
        Returns:
            İşlenmiş yüz görüntüsü
        """
        try:
            if face_img is None or face_img.size == 0:
                return None
                
            # Boyutlandır - daha büyük boyut kullan
            face_img = cv2.resize(face_img, (64, 64))
            
            # Griye çevir (daha tutarlı sonuçlar)
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Kontrast geliştirme
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))
            enhanced_face = clahe.apply(gray_face)
            
            # Gürültü azaltma - daha hassas
            denoised = cv2.GaussianBlur(enhanced_face, (3, 3), 0)
            
            # Histogram eşitleme - kontrastı iyileştir
            equalized = cv2.equalizeHist(denoised)
            
            # Normalize et
            normalized = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX)
            
            # BGR formatına geri dön
            processed_face = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            
            return processed_face
            
        except Exception as e:
            logger.error(f"Yüz ön işleme hatası: {str(e)}")
            return face_img  # Hata durumunda orijinal görüntüyü döndür
    
    def analyze_face_regions(self, face_img):
        """
        Yüzün farklı bölgelerini analiz ederek duygu tespitini iyileştirir
        Args:
            face_img: İşlenmiş yüz görüntüsü
        Returns:
            Bölgesel analiz skorları
        """
        if face_img is None or face_img.size == 0:
            return {}
        
        h, w = face_img.shape[0], face_img.shape[1]
        region_scores = {}
        
        # Her duygu için özel bölgelere bak
        for emotion, regions in self.face_regions.items():
            scores = []
            for x1_ratio, x2_ratio, y1_ratio, y2_ratio in regions:
                # Koordinatları hesapla
                x1, y1 = int(w * x1_ratio), int(h * y1_ratio)
                x2, y2 = int(w * x2_ratio), int(h * y2_ratio)
                
                # Bölgeyi kes
                region = face_img[y1:y2, x1:x2]
                
                # Bölge analizini yap (basit kontrast ve kenar analizi)
                if region.size > 0:
                    # Gri skalaya çevir
                    if len(region.shape) > 2:
                        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    else:
                        region_gray = region
                        
                    # Kenar tespiti
                    edges = cv2.Canny(region_gray, 50, 150)
                    edge_ratio = np.sum(edges > 0) / edges.size
                    
                    # Histogram analizi
                    hist = cv2.calcHist([region_gray], [0], None, [8], [0, 256])
                    hist_std = np.std(hist) / np.mean(hist) if np.mean(hist) > 0 else 0
                    
                    # Bölgenin kontrast ve kenar skorunu hesapla
                    region_score = edge_ratio * 2 + hist_std
                    scores.append(region_score)
            
            # Bölge skorlarının ortalaması
            region_scores[emotion] = np.mean(scores) if scores else 0
                    
        return region_scores
    
    def analyze_emotion(self, face_img, face_id=None):
        """
        Gelişmiş duygu analizi - bölgesel ve zamansal özelliklerle
        Args:
            face_img: İşlenmiş yüz görüntüsü
            face_id: Yüz kimliği (geçmiş duygular için)
        Returns:
            {"emotion": str, "confidence": float} formatında duygu bilgisi
        """
        try:
            # Bellek temizleme kontrolü
            self._cleanup_old_data()
            
            if face_img is None or face_img.size == 0:
                return {"emotion": "unknown", "confidence": 0.0}
                
            # Boyut kontrolü
            if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                return {"emotion": "unknown", "confidence": 0.0}
            
            # Duygu soğutma dönemini kontrol et
            if face_id in self.emotion_cooldown and self.emotion_cooldown[face_id] > 0:
                # Soğuma dönemindeyiz, önceki duyguyu kullan
                self.emotion_cooldown[face_id] -= 1
                if face_id in self.emotion_stability:
                    prev_emotion = self.emotion_stability[face_id]["emotion"]
                    prev_conf = self.emotion_stability[face_id]["confidence"]
                    return {
                        "emotion": self.get_emotion_name(prev_emotion),
                        "confidence": prev_conf
                    }
            
            # Bölgesel analizi yap
            region_scores = self.analyze_face_regions(face_img)
            
            # DeepFace analizi için önbellekleme
            results = {}
            if self.enable_caching:
                # Görüntü hash'ini hesapla
                img_hash = self._compute_image_hash(face_img)
                
                if img_hash and img_hash in self.result_cache:
                    # Önbellekten sonuç al
                    results = self.result_cache[img_hash].copy()
                    self.cache_hit_count += 1
                    logger.debug("Önbellek hit: Duygu analizi önbellekten alındı")
                else:
                    # DeepFace ile analiz yap
                    results = self._analyze_with_models(face_img)
                    self.cache_miss_count += 1
                    
                    # Sonucu önbelleğe ekle
                    if img_hash:
                        self.result_cache[img_hash] = results.copy()
            else:
                # Önbellekleme devre dışı, doğrudan analiz yap
                results = self._analyze_with_models(face_img)
            
            # Sonuçları al
            if results and "emotion_scores" in results:
                # Duygu skorlarını al
                emotion_scores = results["emotion_scores"]
                logger.debug(f"Ham duygu skorları: {emotion_scores}")
                
                # Bölgesel skorlarla duygu skorlarını birleştir
                for emotion, score in emotion_scores.items():
                    if emotion in region_scores:
                        # Bölgesel skoru duygu skoruna entegre et (ağırlıklı ortalama)
                        region_weight = 0.3  # Bölgesel skorların ağırlığı
                        score_weight = 1.0 - region_weight
                        
                        # Normalize edilmiş bölge skoru
                        norm_region_score = region_scores[emotion] * 100
                        
                        # Ağırlıklı ortalama
                        emotion_scores[emotion] = (score * score_weight + norm_region_score * region_weight)
                
                # Kişi bazlı adaptif kalibrasyon uygula
                if face_id is not None:
                    emotion_scores = self._apply_person_calibration(face_id, emotion_scores)
                
                # Zamansal ve benzerlik tabanlı kalibrasyon
                calibrated_scores = self._advanced_calibrate_emotions(emotion_scores, face_id)
                
                # En yüksek skora sahip duyguyu bul
                dominant_emotion = max(calibrated_scores, key=calibrated_scores.get)
                
                # Güven skorunu 0-1 aralığına normalize et (önce 0-100 aralığına)
                emotion_score = calibrated_scores[dominant_emotion]
                normalized_score = min(100.0, max(0.0, emotion_score)) / 100.0 
                
                # Düşük güven skoru kontrolü
                base_threshold = self.emotion_confidence_threshold
                emotion_threshold = self.emotion_calibration.get(dominant_emotion, {}).get("threshold", base_threshold)
                
                # Geçici duygu skorlarını güncelle
                self._update_temporal_scores(face_id, calibrated_scores, normalized_score, dominant_emotion)
                
                # Adaptif kalibrasyonu güncelle
                if face_id is not None:
                    self._update_person_calibration(face_id, dominant_emotion, normalized_score)
                
                if normalized_score < emotion_threshold:
                    logger.debug(f"Düşük duygu güven skoru: {normalized_score} < {emotion_threshold}")
                    # Düşük güvenli duygular için duygu kararlılığını kontrol et
                    # Eğer kararlı bir duygu varsa, düşük güvenli anlık duygu yerine onu kullan
                    stable_info = self._get_stable_emotion(face_id)
                    if stable_info["stability"] > 0.5:
                        return {
                            "emotion": self.get_emotion_name(stable_info["emotion"]), 
                            "confidence": stable_info["confidence"]
                        }
                    return {"emotion": self.get_emotion_name("uncertain"), "confidence": normalized_score}
                
                # Duygu kararlılığını ve benzerliğini kontrol et
                advanced_emotion, advanced_confidence = self._advanced_emotion_stability(
                    face_id, dominant_emotion, normalized_score, calibrated_scores
                )
                
                # Soğuma dönemini ayarla
                if advanced_emotion != dominant_emotion:
                    self.emotion_cooldown[face_id] = self.cooldown_frames
                
                # Kararlı duygu durumunu sakla
                if face_id is not None:
                    self.emotion_stability[face_id] = {
                        "emotion": advanced_emotion,
                        "confidence": advanced_confidence
                    }
                
                logger.info(f"Tespit edilen duygu: {advanced_emotion}, güven: {advanced_confidence:.2f}")
                return {
                    "emotion": self.get_emotion_name(advanced_emotion), 
                    "confidence": advanced_confidence
                }
            else:
                return {"emotion": self.get_emotion_name("unknown"), "confidence": 0.0}

        except Exception as e:
            logger.error(f"Duygu analiz hatası: {str(e)}", exc_info=True)
            return {"emotion": self.get_emotion_name("neutral"), "confidence": 0.5}
    
    def _analyze_with_models(self, face_img):
        """
        Farklı duygu analiz modellerinden gelen sonuçları birleştirir
        Args:
            face_img: Analiz edilecek yüz görüntüsü
        Returns:
            Birleştirilmiş analiz sonuçları
        """
        results = {"emotion_scores": {}}
        start_time = time.time()
        
        try:
            # DeepFace modeli (her zaman etkin)
            if self.models["deepface"]["enabled"]:
                with self.emotion_lock:
                    deepface_result = DeepFace.analyze(
                        face_img, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        detector_backend=self.models["deepface"]["backend"]
                    )
                
                if deepface_result and len(deepface_result) > 0:
                    emotion_scores = deepface_result[0]["emotion"]
                    weight = self.models["deepface"]["weight"]
                    
                    # Skorları ağırlıklarıyla birlikte ekle
                    for emotion, score in emotion_scores.items():
                        results["emotion_scores"][emotion] = score * weight
            
            # Diğer modeller burada eklenebilir...
            # Örnek: TensorFlow veya PyTorch tabanlı özel model
            # if self.models.get("custom_model", {}).get("enabled", False):
            #     custom_result = self._analyze_with_custom_model(face_img)
            #     weight = self.models["custom_model"]["weight"]
            #     for emotion, score in custom_result.items():
            #         if emotion in results["emotion_scores"]:
            #             results["emotion_scores"][emotion] += score * weight
            #         else:
            #             results["emotion_scores"][emotion] = score * weight
            
            # İşlem süresini logla
            process_time = time.time() - start_time
            logger.debug(f"Çoklu model duygu analizi süresi: {process_time:.2f} saniye")
            
            # Sonuçlar boşsa varsayılan değerler
            if not results["emotion_scores"]:
                results["emotion_scores"] = {
                    "neutral": 70.0,
                    "happy": 10.0, 
                    "sad": 5.0,
                    "angry": 5.0,
                    "fear": 3.0,
                    "surprise": 5.0,
                    "disgust": 2.0
                }
                
        except Exception as e:
            logger.error(f"Model analizi hatası: {str(e)}")
            # Hata durumunda varsayılan değerler
            results["emotion_scores"] = {
                "neutral": 90.0,
                "happy": 2.0, 
                "sad": 2.0,
                "angry": 2.0,
                "fear": 2.0,
                "surprise": 1.0,
                "disgust": 1.0
            }
        
        return results
    
    def _apply_person_calibration(self, face_id, emotion_scores):
        """
        Kişi bazlı kalibrasyon değerlerini uygular
        Args:
            face_id: Yüz kimliği
            emotion_scores: Duygu skorları
        Returns:
            Kişiye özel kalibre edilmiş skorlar
        """
        if face_id not in self.person_calibration:
            # Kişi için kalibrasyon verisi yoksa orijinal skorları döndür
            return emotion_scores
        
        calibrated = emotion_scores.copy()
        person_calib = self.person_calibration[face_id]
        
        # Her duygu için kişi bazlı kalibrasyonu uygula
        for emotion in calibrated:
            if emotion in person_calib:
                # Kişi bazlı boost faktörünü uygula
                boost_factor = person_calib[emotion]
                calibrated[emotion] *= boost_factor
        
        return calibrated
    
    def _update_person_calibration(self, face_id, dominant_emotion, confidence):
        """
        Kişi bazlı kalibrasyon değerlerini günceller
        Args:
            face_id: Yüz kimliği
            dominant_emotion: Tespit edilen baskın duygu
            confidence: Duygu güven skoru
        """
        if face_id is None:
            return
            
        # Kişi kalibrasyonunu başlat
        if face_id not in self.person_calibration:
            self.person_calibration[face_id] = {
                "angry": 1.0,
                "disgust": 1.0,
                "fear": 1.0,
                "happy": 1.0,
                "sad": 1.0,
                "surprise": 1.0,
                "neutral": 1.0
            }
        
        # Sadece yüksek güvenli tanımlamaları öğrenme için kullan
        if confidence < 0.6:
            return
            
        # Tespit edilen duyguya göre kişi bazlı kalibrasyon değerini ayarla
        # Doğru tespit edilen duyguyu hafifçe güçlendir
        current_boost = self.person_calibration[face_id].get(dominant_emotion, 1.0)
        # Maksimum 1.5 kat boost (yüzde 50 artış)
        new_boost = min(1.5, current_boost + (self.adaptation_rate * (1.0 - current_boost)))
        self.person_calibration[face_id][dominant_emotion] = new_boost
        
        # Diğer duyguları hafifçe düşür (dengeli bir toplam için)
        for emotion in self.person_calibration[face_id]:
            if emotion != dominant_emotion:
                current_val = self.person_calibration[face_id][emotion]
                # Minimum 0.7 kat (yüzde 30 azalma)
                new_val = max(0.7, current_val - (self.adaptation_rate * 0.1))
                self.person_calibration[face_id][emotion] = new_val
    
    def get_emotion_name(self, emotion_key):
        """
        Duygu anahtarını döndürür (istenirse Türkçe karşılığını kullanabilir)
        Args:
            emotion_key: Duygu kodu (angry, happy, vb.)
        Returns:
            Duygu adı (varsayılan olarak İngilizce)
        """
        # Türkçe duygu isimlerini kullanmak için:
        # return self.emotion_tr.get(emotion_key, emotion_key)
        
        # İngilizce kullanmak için:
        return emotion_key
            
    def _advanced_calibrate_emotions(self, emotion_scores, face_id=None):
        """
        Geliştirilmiş duygu skoru kalibrasyonu - zamansal ve benzerlik bazlı
        Args:
            emotion_scores: Orijinal duygu skorları
            face_id: Yüz kimliği (varsa)
        Returns:
            Kalibre edilmiş duygu skorları
        """
        calibrated = {}
        
        # Eğer zamansal skorlar yoksa başlat
        if face_id is not None and face_id not in self.emotion_temporal_scores:
            self.emotion_temporal_scores[face_id] = {}
            # Başlangıçta tüm duygular için ortalama skor 0
            for emotion in emotion_scores.keys():
                self.emotion_temporal_scores[face_id][emotion] = 0.0
        
        # Önceki stabiliteyi kontrol et
        stable_info = self._get_stable_emotion(face_id) if face_id is not None else {"emotion": None, "stability": 0}
        stable_emotion = stable_info["emotion"]
        stable_stability = stable_info["stability"]
        
        # Mevcut dominant duyguyu bul
        if emotion_scores:
            current_dominant = max(emotion_scores, key=emotion_scores.get)
        else:
            current_dominant = "neutral"
        
        for emotion, score in emotion_scores.items():
            # Temel kalibrasyon faktörleri
            boost = self.emotion_calibration.get(emotion, {}).get("boost", 1.0)
            priority = self.emotion_calibration.get(emotion, {}).get("priority", 0)
            
            # Skorun önceliğe dayalı ek boost faktörü
            priority_boost = 1.0 + (priority * 0.05)  # Her öncelik seviyesi için %5 boost
            
            # Eğer stabil bir duygu varsa ve şu anki duygu ona benziyorsa boost ver
            if stable_emotion and stable_stability > 0.5:
                if emotion == stable_emotion:
                    # Stabil duyguysa ekstra boost
                    stable_boost = 1.0 + (stable_stability * 0.2)  # En fazla %20 boost
                    boost *= stable_boost
                elif emotion in self.emotion_similarity and stable_emotion in self.emotion_similarity[emotion]:
                    # Stabil duyguya benzer bir duyguysa kısmen boost
                    similarity = self.emotion_similarity[emotion][stable_emotion]
                    similar_boost = 1.0 + (similarity * 0.1)  # En fazla %10 boost
                    boost *= similar_boost
            
            # Zamansal faktörü hesapla - önceki skorlarla şimdi arasındaki korelasyon
            temporal_factor = 1.0
            if face_id is not None:
                # Önceki ortalama skorlarla mevcut skor arasındaki ilişki
                prev_avg = self.emotion_temporal_scores[face_id].get(emotion, 0)
                
                # Eğer önceki ortalama yüksekse ve şimdiki skor da yüksekse doğrulayıcı boost ver
                if prev_avg > 30 and score > 40:
                    temporal_factor = 1.1  # %10 boost
                # Eğer önceki ortalama düşükse ve şimdi çok yüksekse şüpheli - azalt
                elif prev_avg < 20 and score > 60:
                    temporal_factor = 0.9  # %10 azalt
            
            # Benzer duygular için çakışma kontrolü
            # Eğer bu duygu, dominant duyguya çok benziyorsa ve baskın değilse azalt
            if emotion != current_dominant and emotion in self.emotion_similarity:
                if current_dominant in self.emotion_similarity[emotion]:
                    similarity = self.emotion_similarity[emotion][current_dominant]
                    if similarity > 0.5 and score < emotion_scores[current_dominant]:
                        # Çok benzer ama daha zayıf bir duygu - azalt
                        similarity_penalty = 1.0 - (similarity * 0.2)  # En fazla %20 azalt
                        boost *= similarity_penalty
            
            # Tüm faktörleri birleştir
            final_boost = boost * priority_boost * temporal_factor
            
            # Son kalibrasyon değeri ile çarparak yeni skor hesapla
            calibrated[emotion] = min(100.0, score * final_boost)  # 100'den büyük olmasın
            
        return calibrated
            
    def _update_temporal_scores(self, face_id, calibrated_scores, confidence, dominant_emotion):
        """
        Zamansal duygu skorlarını günceller
        Args:
            face_id: Yüz kimliği
            calibrated_scores: Kalibre edilmiş duygu skorları
            confidence: Genel güven skoru
            dominant_emotion: Baskın duygu
        """
        if face_id is None:
            return
            
        # Yüz için zamansal skorları başlat
        if face_id not in self.emotion_temporal_scores:
            self.emotion_temporal_scores[face_id] = {}
            
        # Her duygu için hareketli ortalama hesapla
        decay = 0.8  # Önceki değerlerin ağırlığı
        for emotion, score in calibrated_scores.items():
            if emotion in self.emotion_temporal_scores[face_id]:
                # Hareketli ortalama güncelleme
                prev_score = self.emotion_temporal_scores[face_id][emotion]
                new_score = prev_score * decay + score * (1 - decay)
                self.emotion_temporal_scores[face_id][emotion] = new_score
            else:
                # İlk değeri ata
                self.emotion_temporal_scores[face_id][emotion] = score
                
        # Yüz için duygu geçmişini güncelle
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = deque(maxlen=self.history_max_size)
            
        # Mevcut duyguyu geçmişe ekle
        self.emotion_history[face_id].append({
            "emotion": dominant_emotion,
            "confidence": confidence,
            "scores": calibrated_scores.copy(),
            "timestamp": time.time()
        })
    
    def _get_stable_emotion(self, face_id):
        """
        Yüz için en kararlı duyguyu bulur
        Args:
            face_id: Yüz kimliği
        Returns:
            Kararlı duygu bilgisi (emotion, confidence, stability)
        """
        if face_id is None or face_id not in self.emotion_history:
            return {"emotion": "neutral", "confidence": 0.5, "stability": 0.0}
            
        # Son N duygusal değeri analiz et
        emotion_counts = {}
        emotion_confs = {}
        
        # Geçmiş kayıtları daha yeni olanlara daha fazla ağırlık vererek analiz et
        weights = np.linspace(0.7, 1.0, len(self.emotion_history[face_id]))
        total_weight = sum(weights)
        
        for i, record in enumerate(self.emotion_history[face_id]):
            emotion = record["emotion"]
            conf = record["confidence"]
            weight = weights[i]
            
            if emotion in emotion_counts:
                emotion_counts[emotion] += weight
                emotion_confs[emotion] += conf * weight
            else:
                emotion_counts[emotion] = weight
                emotion_confs[emotion] = conf * weight
                
        # Emotion count yoksa nötr döndür
        if not emotion_counts:
            return {"emotion": "neutral", "confidence": 0.5, "stability": 0.0}
                
        # En çok görülen duygu
        stable_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Kararlılık = en çok görülen duygunun toplam ağırlığı / toplam ağırlık
        stability = emotion_counts[stable_emotion] / total_weight if total_weight > 0 else 0
        
        # Ortalama güven skoru
        avg_conf = emotion_confs[stable_emotion] / emotion_counts[stable_emotion] if emotion_counts[stable_emotion] > 0 else 0
        
        return {
            "emotion": stable_emotion,
            "confidence": min(1.0, avg_conf), 
            "stability": stability
        }
        
    def _advanced_emotion_stability(self, face_id, current_emotion, confidence, all_scores):
        """
        Gelişmiş duygu kararlılığı kontrolü
        Args:
            face_id: Yüz kimliği
            current_emotion: Mevcut duygu
            confidence: Mevcut güven skoru
            all_scores: Tüm duygu skorları
        Returns:
            (nihai_duygu, nihai_güven) tuple
        """
        if face_id is None:
            return current_emotion, confidence
            
        # Kararlı duyguyu al
        stable_info = self._get_stable_emotion(face_id)
        stable_emotion = stable_info["emotion"]
        stable_confidence = stable_info["confidence"]
        stability = stable_info["stability"]
        
        # Skorları sırala
        sorted_emotions = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        # En yüksek iki duygu arasındaki fark
        emotion_margin = 0.0
        if len(sorted_emotions) >= 2:
            emotion_margin = (sorted_emotions[0][1] - sorted_emotions[1][1]) / 100.0
            
        # Çok benzer skorlu duygular arasındaki çakışma kontrolü
        if emotion_margin < 0.1:  # İlk iki duygu arasında %10'dan az fark varsa
            # İkinci duygu ile birinci duygu arasındaki benzerlik
            second_emotion = sorted_emotions[1][0]
            
            # Eğer ikinci duygu kararlı duyguysa ve kararlılık yüksekse, onu kullan
            if second_emotion == stable_emotion and stability > 0.6:
                return stable_emotion, stable_confidence
                
            # Eğer mevcut duygu ile ikinci duygu arasında benzerlik yüksekse
            if current_emotion in self.emotion_similarity and second_emotion in self.emotion_similarity[current_emotion]:
                similarity = self.emotion_similarity[current_emotion][second_emotion]
                
                # Çok benzer duygular ve kararlı duygu varsa karar ver
                if similarity > 0.6 and stable_emotion in [current_emotion, second_emotion]:
                    return stable_emotion, stable_confidence
        
        # Kararlı duygumuz var ve güven skorumuz yüksekse kararlı duyguda kal
        if stable_emotion and stability > 0.7 and confidence < 0.8:
            return stable_emotion, stable_confidence
            
        # Duygular arasında ani değişimleri yumuşat:
        # 1. Eğer mevcut duygu, stabil duygudan çok farklıysa ve güven yeterince yüksek değilse
        if (stable_emotion != current_emotion and 
            stability > 0.5 and 
            confidence < 0.7):
                
            # Stabil duygu ile mevcut duygu arasındaki benzerliği kontrol et
            similarity = 0.0
            if current_emotion in self.emotion_similarity and stable_emotion in self.emotion_similarity[current_emotion]:
                similarity = self.emotion_similarity[current_emotion][stable_emotion]
                
            # Eğer çok farklı duygularsa (düşük benzerlik) ve stabil duygumuz güçlüyse
            if similarity < 0.3 and stability > 0.6:
                # Stabil duyguyu kullan
                return stable_emotion, stable_confidence
                
        # Diğer durumlarda mevcut duyguyu kullan
        return current_emotion, confidence
    
    def get_cache_stats(self):
        """
        Önbellek istatistiklerini döndürür
        Returns:
            Önbellek kullanım istatistikleri
        """
        if not self.enable_caching:
            return {"enabled": False}
            
        total = self.cache_hit_count + self.cache_miss_count
        hit_rate = (self.cache_hit_count / total * 100) if total > 0 else 0
        
        return {
            "enabled": True,
            "size": len(self.result_cache),
            "max_size": self.cache_size,
            "hits": self.cache_hit_count,
            "misses": self.cache_miss_count,
            "hit_rate": hit_rate,
            "total_queries": total
        }
