import cv2
import numpy as np
import time
import logging
import threading
from collections import deque
from deepface import DeepFace

# Logging yapılandırma
logger = logging.getLogger("emotion-analyzer")

class EmotionAnalyzer:
    """Duygu analizi için geliştirilmiş sınıf"""
    
    def __init__(self, confidence_threshold=0.30):
        """
        Emotion Analyzer sınıfını başlatır
        Args:
            confidence_threshold: Duygu tespiti için minimum güven skoru
        """
        self.emotion_confidence_threshold = confidence_threshold
        self.emotion_lock = threading.Lock()  # Thread güvenliği için
        
        # Duygu sınıflandırıcısını kalibre etme değerleri - daha hassas kalibre
        self.emotion_calibration = {
            "angry":    {"threshold": 0.35, "boost": 1.50, "decay": 0.80, "priority": 3},
            "disgust":  {"threshold": 0.45, "boost": 1.10, "decay": 0.90, "priority": 1},
            "fear":     {"threshold": 0.38, "boost": 1.40, "decay": 0.80, "priority": 2},
            # Happy için eşiği ve boost’u düşürdük, decay’i artırdık
            "happy":    {"threshold": 0.35, "boost": 1.10, "decay": 0.90, "priority": 2},
            # Sad için eşiği ve boost’u düşürdük, decay’i artırdık
            "sad":      {"threshold": 0.42, "boost": 1.05, "decay": 0.90, "priority": 1},
            "surprise": {"threshold": 0.32, "boost": 1.40, "decay": 0.75, "priority": 3},
            "neutral":  {"threshold": 0.25, "boost": 0.95, "decay": 0.70, "priority": 0}
        }
        
        # Duygular arası geçişleri düzenleyen benzerlik matrisi (1: Çok benzer, 0: Hiç benzer değil)
        self.emotion_similarity = {
            "angry":    {"disgust": 0.6, "fear": 0.3, "happy": 0.0, "sad": 0.4, "surprise": 0.2, "neutral": 0.1},
            "disgust":  {"angry": 0.6, "fear": 0.4, "happy": 0.0, "sad": 0.5, "surprise": 0.1, "neutral": 0.2},
            "fear":     {"angry": 0.3, "disgust": 0.4, "happy": 0.0, "sad": 0.5, "surprise": 0.7, "neutral": 0.2},
            "happy":    {"angry": 0.0, "disgust": 0.0, "fear": 0.0, "sad": 0.0, "surprise": 0.4, "neutral": 0.3},
            "sad":      {"angry": 0.4, "disgust": 0.5, "fear": 0.5, "happy": 0.0, "surprise": 0.1, "neutral": 0.5},
            "surprise": {"angry": 0.2, "disgust": 0.1, "fear": 0.7, "happy": 0.4, "sad": 0.1, "neutral": 0.2},
            "neutral":  {"angry": 0.1, "disgust": 0.2, "fear": 0.2, "happy": 0.3, "sad": 0.5, "surprise": 0.2}
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
            "angry":    [(0.3, 0.7, 0.0, 0.6)],
            "disgust":  [(0.2, 0.8, 0.4, 0.8)],
            "fear":     [(0.2, 0.8, 0.1, 0.6)],
            "happy":    [(0.2, 0.8, 0.5, 0.9)],
            "sad":      [(0.3, 0.7, 0.3, 0.8)],
            "surprise": [(0.2, 0.8, 0.1, 0.7)]
        }
        
        # Son duygu geçmişi, stabil tespitler için
        self.emotion_history = {}
        self.emotion_temporal_scores = {}
        self.history_max_size = 7  # Daha uzun tarihçe
        self.emotion_stability = {}
        
        # Duygu geçiş soğutma süresi (ani değişimleri engeller)
        self.emotion_cooldown = {}
        self.cooldown_frames = 2
        
        logger.info("Geliştirilmiş duygu analizi modülü başlatıldı")
        
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
                
            # Boyutlandır
            face_img = cv2.resize(face_img, (64, 64))
            # Griye çevir
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))
            enhanced_face = clahe.apply(gray_face)
            # Gürültü azaltma
            denoised = cv2.GaussianBlur(enhanced_face, (3, 3), 0)
            # Histogram eşitleme
            equalized = cv2.equalizeHist(denoised)
            # Normalize et
            normalized = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX)
            # BGR formatına geri dön
            processed_face = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            return processed_face

        except Exception as e:
            logger.error(f"Yüz ön işleme hatası: {str(e)}")
            return face_img

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
        
        h, w = face_img.shape[:2]
        region_scores = {}
        
        for emotion, regions in self.face_regions.items():
            scores = []
            for x1r, x2r, y1r, y2r in regions:
                x1, y1 = int(w * x1r), int(h * y1r)
                x2, y2 = int(w * x2r), int(h * y2r)
                region = face_img[y1:y2, x1:x2]
                if region.size > 0:
                    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim==3 else region
                    edges = cv2.Canny(region_gray, 50, 150)
                    edge_ratio = np.sum(edges>0)/edges.size
                    hist = cv2.calcHist([region_gray],[0],None,[8],[0,256])
                    hist_std = np.std(hist)/np.mean(hist) if np.mean(hist)>0 else 0
                    region_score = edge_ratio*2 + hist_std
                    scores.append(region_score)
            region_scores[emotion] = np.mean(scores) if scores else 0
        
        return region_scores

    def analyze_emotion(self, face_img, face_id=None):
        """
        Gelişmiş duygu analizi - bölgesel ve zamansal özelliklerle
        Args:
            face_img: İşlenmiş yüz görüntüsü
            face_id:   Yüz kimliği (geçmiş duygular için)
        Returns:
            {"emotion": str, "confidence": float}
        """
        try:
            if face_img is None or face_img.size == 0:
                return {"emotion":"unknown","confidence":0.0}
            if face_img.shape[0]<10 or face_img.shape[1]<10:
                return {"emotion":"unknown","confidence":0.0}

            # Soğutma kontrolü
            if face_id in self.emotion_cooldown and self.emotion_cooldown[face_id]>0:
                self.emotion_cooldown[face_id]-=1
                if face_id in self.emotion_stability:
                    prev = self.emotion_stability[face_id]
                    return {"emotion":self.get_emotion_name(prev["emotion"]),"confidence":prev["confidence"]}

            region_scores = self.analyze_face_regions(face_img)
            start = time.time()
            with self.emotion_lock:
                result = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
            logger.debug(f"Duygu analizi süresi: {time.time()-start:.2f}s")

            if result and len(result)>0:
                emotion_scores = result[0]["emotion"]
                logger.debug(f"Ham duygu skorları: {emotion_scores}")

                # Bölgesel katkı
                for emotion, score in emotion_scores.items():
                    if emotion in region_scores:
                        # Bölgesel skoru zayıflat: happy/sad için 0.15, diğerleri 0.3
                        if emotion in ("happy","sad"):
                            region_weight = 0.15
                        else:
                            region_weight = 0.3
                        score_weight = 1.0 - region_weight
                        norm_region = region_scores[emotion]*100
                        emotion_scores[emotion] = score*score_weight + norm_region*region_weight

                # Kalibrasyon
                calibrated = self._advanced_calibrate_emotions(emotion_scores, face_id)
                dominant = max(calibrated, key=calibrated.get)
                raw_score = calibrated[dominant]
                norm_score = min(100.0,max(0.0,raw_score))/100.0

                # Güncelle
                base_thr = self.emotion_confidence_threshold
                emo_thr = self.emotion_calibration.get(dominant,{}).get("threshold",base_thr)
                self._update_temporal_scores(face_id, calibrated, norm_score, dominant)

                if norm_score < emo_thr:
                    stable = self._get_stable_emotion(face_id)
                    if stable["stability"]>0.5:
                        return {"emotion":self.get_emotion_name(stable["emotion"]),"confidence":stable["confidence"]}
                    return {"emotion":self.get_emotion_name("uncertain"),"confidence":norm_score}

                adv_emo, adv_conf = self._advanced_emotion_stability(face_id, dominant, norm_score, calibrated)
                if adv_emo!=dominant:
                    self.emotion_cooldown[face_id] = self.cooldown_frames

                logger.info(f"Tespit edilen duygu: {adv_emo}, güven: {adv_conf:.2f}")
                return {"emotion":self.get_emotion_name(adv_emo),"confidence":adv_conf}

            return {"emotion":self.get_emotion_name("unknown"),"confidence":0.0}

        except Exception as e:
            logger.error(f"DeepFace analiz hatası: {str(e)}")
            return {"emotion":self.get_emotion_name("neutral"),"confidence":0.5}

    def get_emotion_name(self, key):
        return key

    # ... (Diğer metodlar: _advanced_calibrate_emotions, _update_temporal_scores,
    #     _get_stable_emotion, _advanced_emotion_stability aynı kalacak)

            
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
