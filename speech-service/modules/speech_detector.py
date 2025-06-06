"""
Speech detection module - ağız hareketi tespiti yoluyla konuşma tespiti yapan modül
"""
import numpy as np
from collections import defaultdict, deque
import logging

logger = logging.getLogger("speech-detector")

class SpeechDetector:
    """Ağız hareketi tespiti yoluyla konuşma tespiti yapan sınıf"""
    
    def __init__(self, 
                 variation_threshold=0.03, 
                 confidence_threshold=0.35, 
                 cooldown_frames=3,
                 history_length=20,
                 adaptation_rate=0.08):
        """
        SpeechDetector sınıfını başlatır
        
        Args:
            variation_threshold (float): Minimum ağız hareketi varyasyon eşiği
            confidence_threshold (float): Konuşma güven puanı eşiği
            cooldown_frames (int): Durum değişikliği için bekleme karesi sayısı
            history_length (int): Tarihçede tutulacak kare sayısı
            adaptation_rate (float): Kişiye özel adaptasyon hızı
        """
        # Parametreler için ayarlar
        self.min_variation_threshold = variation_threshold
        self.speaking_confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames
        
        # Kişi bazlı değişkenler 
        self.mouth_history = defaultdict(list)
        self.history_length = history_length
        self.person_thresholds = {}
        self.speaking_confidence = defaultdict(float)
        self.speaking_state = defaultdict(bool)
        self.transition_cooldown = defaultdict(int)
        
        # Konuşma süresi takibi için değişkenler
        self.speaking_start_times = {}
        self.speaking_durations = defaultdict(float)
        
        # Konuşma tespiti için değişkenler
        self.mouth_motion_history = defaultdict(lambda: deque(maxlen=12))
        self.last_features = defaultdict(dict)
        
        # Adaptasyon parametreleri
        self.adaptation_rate = adaptation_rate
        
        # Gelişmiş özellikler
        self.sustained_speaking_bonus = 0.03  # Sürekli konuşma durumlarında uygulanacak bonus
        self.motion_weight = 0.35  # Hareket tespitinin ağırlığı
        self.opening_weight = 0.35  # Ağız açıklığının ağırlığı
        self.distance_weight = 0.20  # Dudaklar arası mesafenin ağırlığı
        self.variation_weight = 0.25  # Değişim analizinin ağırlığı
        
        logger.info("Konuşma tespit modülü başlatıldı")
        
    def calc_mouth_opening(self, landmarks):
        """
        Verilen landmarks'dan ağız açıklık özelliklerini hesaplar
        
        Args:
            landmarks (list): 68 yüz noktası (x,y koordinatları art arda dizilmiş)
            
        Returns:
            dict: Ağız özellikleri içeren sözlük
        """
        # Landmark verisi float listesi olarak gelir, yeniden yapılandır
        if not landmarks or len(landmarks) < 136:  # 68 nokta * 2 (x,y)
            return {
                'normalized_opening': 0.0,
                'horizontal_width': 0.0,
                'avg_lip_distance': 0.0,
                'symmetry': 1.0,
                'mouth_aspect_ratio': 0.0
            }
            
        # Landmark listesinden noktaları çıkar
        points = []
        for i in range(0, len(landmarks), 2):
            if i+1 < len(landmarks):
                points.append((landmarks[i], landmarks[i+1]))
        
        # Ağız noktaları - indeksler (dlib 68 nokta formatına göre)
        inner_lip_indices = list(range(60, 68))  # İç dudak noktaları (60-67)
        top_middle_index = 62  # Üst dudak ortası
        bottom_middle_index = 66  # Alt dudak ortası
        left_corner_index = 48  # Sol dudak köşesi
        right_corner_index = 54  # Sağ dudak köşesi
        
        # Dikey açıklık - orta noktalarda
        if top_middle_index < len(points) and bottom_middle_index < len(points):
            vertical_opening = abs(points[top_middle_index][1] - points[bottom_middle_index][1])
        else:
            vertical_opening = 0
        
        # Yatay genişlik
        if left_corner_index < len(points) and right_corner_index < len(points):
            horizontal_width = abs(points[left_corner_index][0] - points[right_corner_index][0])
        else:
            horizontal_width = 1  # Sıfıra bölünmeyi önlemek için
        
        # Ağız Görünüm Oranı (MAR - Mouth Aspect Ratio) hesapla
        # MAR = Dikey açıklık / Yatay genişlik
        mouth_aspect_ratio = vertical_opening / horizontal_width if horizontal_width > 0 else 0
        
        # Normalize edilmiş ağız açıklığı
        normalized_opening = vertical_opening / (horizontal_width * 0.5) if horizontal_width > 0 else 0
        
        # Dudaklar arası mesafe hesapla
        top_lip_points = [51, 52, 53]  # Üst dudak noktaları
        bottom_lip_points = [57, 58, 59]  # Alt dudak noktaları
        
        distances = []
        for i in range(min(len(top_lip_points), len(bottom_lip_points))):
            if (top_lip_points[i] < len(points) and bottom_lip_points[i] < len(points)):
                top_pt = points[top_lip_points[i]]
                bottom_pt = points[bottom_lip_points[i]]
                distance = np.sqrt((top_pt[0] - bottom_pt[0])**2 + (top_pt[1] - bottom_pt[1])**2)
                distances.append(distance)
        
        avg_lip_distance = np.mean(distances) if distances else 0
                
        # Dudak simetrisi (sağ ve sol taraf açıklık karşılaştırması)
        left_side_opening = 0
        right_side_opening = 0
        
        if 61 < len(points) and 67 < len(points):
            left_side_opening = abs(points[61][1] - points[67][1])
        
        if 63 < len(points) and 65 < len(points):
            right_side_opening = abs(points[63][1] - points[65][1])
        
        # Simetri hesaplama (1'e yakın değer daha simetrik)
        symmetry = min(left_side_opening, right_side_opening) / max(left_side_opening, right_side_opening) if max(left_side_opening, right_side_opening) > 0 else 1.0
        
        # Sonuçları döndür
        return {
            'normalized_opening': normalized_opening,
            'horizontal_width': horizontal_width,
            'avg_lip_distance': avg_lip_distance,
            'symmetry': symmetry,
            'mouth_aspect_ratio': mouth_aspect_ratio
        }
        
    def detect_speaking(self, face_id, landmarks, current_time=None):
        """
        Konuşma durumunu tespit eder
        
        Args:
            face_id (str): Yüz kimliği
            landmarks (list): Yüz noktaları
            current_time (float, optional): Mevcut zaman damgası
            
        Returns:
            bool: Konuşma durumu (True/False)
        """
        # Ağız özelliklerini hesapla
        mouth_features = self.calc_mouth_opening(landmarks)
        
        # Ağız özelliklerini geçmişe ekle
        self.mouth_history[face_id].append(mouth_features)

        # Geçmişi belirli bir boyutta sınırla
        if len(self.mouth_history[face_id]) > self.history_length:
            self.mouth_history[face_id].pop(0)
            
        # Son kareden beri olan değişimi kaydet
        if face_id in self.last_features:
            # Ağız açıklığındaki değişim
            opening_change = abs(mouth_features['normalized_opening'] - 
                               self.last_features[face_id].get('normalized_opening', 0))
            
            # Dudak mesafe değişimi
            distance_change = abs(mouth_features['avg_lip_distance'] - 
                                self.last_features[face_id].get('avg_lip_distance', 0))
            
            # MAR değişimi (Mouth Aspect Ratio)
            mar_change = abs(mouth_features['mouth_aspect_ratio'] - 
                           self.last_features[face_id].get('mouth_aspect_ratio', 0))
            
            # Tüm değişimleri birleştir (ağırlıklı ortalama)
            total_change = (opening_change * 0.6) + (distance_change * 0.25) + (mar_change * 0.15)
            
            # Değişim geçmişini kaydet
            self.mouth_motion_history[face_id].append(total_change)
        
        # Mevcut değerleri güncelle
        self.last_features[face_id] = mouth_features.copy()

        # Yeterli veri yoksa basit bir kontrol yap
        if len(self.mouth_history[face_id]) < 5:
            # Başlangıç kontrolü
            raw_speaking = mouth_features['normalized_opening'] > 0.3 and mouth_features['avg_lip_distance'] > 3.0
            self.speaking_state[face_id] = raw_speaking
            
            # Konuşma süresini güncelle
            if current_time:
                self.update_speaking_time(face_id, raw_speaking, current_time)
                
            return raw_speaking

        # Kişiye özel eşik değerini oluştur veya güncelle
        if face_id not in self.person_thresholds or len(self.mouth_history[face_id]) % 10 == 0:
            self._update_person_thresholds(face_id)

        # Mevcut değerler
        current_opening = mouth_features['normalized_opening']
        current_distance = mouth_features['avg_lip_distance']
        current_mar = mouth_features['mouth_aspect_ratio']
        
        # Eşik değerleri
        thresholds = self.person_thresholds.get(face_id, {'opening': 0.2, 'distance': 2.0, 'mar': 0.4})
        
        # Hareket tespiti
        motion_detected = False
        strong_motion = False
        if len(self.mouth_motion_history[face_id]) >= 4:
            # Son karelerdeki hareket miktarını analiz et
            recent_motion = list(self.mouth_motion_history[face_id])
            
            # Ortalama hareket ve varyans
            mean_motion = np.mean(recent_motion)
            motion_variance = np.var(recent_motion) if len(recent_motion) > 1 else 0
            
            # Hareket tespit kriterleri
            motion_detected = (mean_motion > self.min_variation_threshold) and (motion_variance > 0.0005)
            
            # Güçlü hareket tespiti
            strong_motion = (mean_motion > self.min_variation_threshold * 2) and (motion_variance > 0.001)

        # Konuşma göstergeleri
        is_open = current_opening > thresholds.get('opening', 0.2)
        has_distance = current_distance > thresholds.get('distance', 2.0)
        good_mar = current_mar > thresholds.get('mar', 0.4)
        
        # Değişim analizi
        recent_frames = min(8, len(self.mouth_history[face_id]))
        recent = self.mouth_history[face_id][-recent_frames:]
        
        # Açıklık değişimi
        openings = [frame['normalized_opening'] for frame in recent]
        opening_std = np.std(openings)
        opening_range = np.max(openings) - np.min(openings)
        
        # MAR değişimi
        mar_values = [frame.get('mouth_aspect_ratio', 0) for frame in recent]
        mar_std = np.std(mar_values)
        
        # Değişim kriterleri
        has_variation = (opening_std > self.min_variation_threshold) or (opening_range > 0.15)
        has_mar_variation = mar_std > 0.02
        
        # Konuşma skoru hesaplama
        speaking_score = (
            self.opening_weight * int(is_open) +            # Ağız açıklığı
            self.distance_weight * int(has_distance) +      # Dudaklar arası mesafe
            self.variation_weight * int(has_variation) +    # Değişim analizi
            self.motion_weight * int(motion_detected) +     # Hareket tespiti
            0.10 * int(good_mar) +                         # Mouth Aspect Ratio
            0.10 * int(has_mar_variation) +                # MAR değişimi
            0.15 * int(strong_motion)                      # Güçlü hareket bonus
        )
        
        # Skoru normalize et
        speaking_score = min(1.0, speaking_score)
        
        # Sürekli konuşma durumlarında ek bonus uygula
        if self.speaking_state[face_id] and speaking_score > 0.4:
            speaking_score += self.sustained_speaking_bonus
        
        # Konuşma güven skorunu güncelle
        if speaking_score > 0.5:
            # Konuşma göstergeleri varsa güven skorunu arttır
            self.speaking_confidence[face_id] = min(1.0, self.speaking_confidence[face_id] + 0.12)
        elif speaking_score < 0.25:
            # Yoksa düşür
            self.speaking_confidence[face_id] = max(0.0, self.speaking_confidence[face_id] - 0.15)
        
        # Mevcut karara karar ver
        raw_speaking_state = self.speaking_confidence[face_id] > self.speaking_confidence_threshold
        
        # Soğuma süreci - ani değişimleri önlemek için
        if raw_speaking_state != self.speaking_state[face_id]:
            if self.transition_cooldown[face_id] >= self.cooldown_frames:
                self.speaking_state[face_id] = raw_speaking_state
                self.transition_cooldown[face_id] = 0
            else:
                self.transition_cooldown[face_id] += 1
        else:
            self.transition_cooldown[face_id] = 0
        
        # Konuşma süresini güncelle
        if current_time:
            self.update_speaking_time(face_id, self.speaking_state[face_id], current_time)
        
        return self.speaking_state[face_id]
    
    def _update_person_thresholds(self, face_id):
        """
        Kişiye özel konuşma eşiklerini günceller
        
        Args:
            face_id (str): Yüz kimliği
        """
        if len(self.mouth_history[face_id]) < 5:
            return
            
        # Geçmiş verilerden istatistikleri hesapla
        openings = [frame['normalized_opening'] for frame in self.mouth_history[face_id]]
        distances = [frame['avg_lip_distance'] for frame in self.mouth_history[face_id]]
        mar_values = [frame.get('mouth_aspect_ratio', 0) for frame in self.mouth_history[face_id]]
        
        # Medyan değerler
        opening_median = np.median(openings)
        distance_median = np.median(distances)
        mar_median = np.median(mar_values)
        
        # Çeyrekler arası aralık (IQR)
        opening_q75 = np.percentile(openings, 75)
        opening_q25 = np.percentile(openings, 25)
        opening_iqr = opening_q75 - opening_q25
        
        # Dinlenme halindeki ağız açıklığı
        resting_opening = np.percentile(openings, 30)
        
        # Dinamik eşikler
        opening_thresh = resting_opening + (opening_iqr * 0.7)
        
        # Mesafe eşiği
        distance_thresh = distance_median * 1.15
        
        # MAR eşiği
        mar_thresh = max(0.3, mar_median * 1.2)
        
        # Yeni eşik değerleri
        new_thresholds = {
            'opening': max(0.15, opening_thresh),
            'distance': max(1.5, distance_thresh),
            'mar': mar_thresh
        }
        
        # Mevcut değerleri al
        current_thresholds = self.person_thresholds.get(face_id, new_thresholds)
        
        # Tedrici olarak güncelle (yumuşak geçiş)
        for key in new_thresholds:
            if key in current_thresholds:
                current_thresholds[key] = (current_thresholds[key] * (1 - self.adaptation_rate) + 
                                          new_thresholds[key] * self.adaptation_rate)
            else:
                current_thresholds[key] = new_thresholds[key]
                
        # Eşik değerlerini güncelle
        self.person_thresholds[face_id] = current_thresholds
        
    def update_speaking_time(self, face_id, is_speaking, current_time):
        """
        Konuşma süresini günceller
        
        Args:
            face_id (str): Yüz kimliği
            is_speaking (bool): Konuşma durumu
            current_time (float): Mevcut zaman damgası
        """
        if is_speaking:
            # Konuşmaya başladıysa başlangıç zamanını kaydet
            if face_id not in self.speaking_start_times:
                self.speaking_start_times[face_id] = current_time
        else:
            # Konuşmayı bitirdiyse süreyi hesapla ve ekle
            if face_id in self.speaking_start_times:
                elapsed = current_time - self.speaking_start_times[face_id]
                # Çok kısa süreli konuşmaları filtrele
                if elapsed > 0.5:  
                    self.speaking_durations[face_id] += elapsed
                del self.speaking_start_times[face_id]
                
    def get_speaking_time(self, face_id):
        """
        Konuşma süresini döndürür
        
        Args:
            face_id (str): Yüz kimliği
            
        Returns:
            float: Toplam konuşma süresi (saniye)
        """
        return self.speaking_durations[face_id]
        
    def clear_face_data(self, face_id):
        """
        Yüze ait verileri temizler
        
        Args:
            face_id (str): Yüz kimliği
        """
        if face_id in self.speaking_start_times:
            del self.speaking_start_times[face_id]
        if face_id in self.speaking_durations:
            del self.speaking_durations[face_id]
        if face_id in self.mouth_history:
            del self.mouth_history[face_id]
        if face_id in self.person_thresholds:
            del self.person_thresholds[face_id]
        if face_id in self.speaking_confidence:
            del self.speaking_confidence[face_id]
        if face_id in self.speaking_state:
            del self.speaking_state[face_id]
        if face_id in self.mouth_motion_history:
            del self.mouth_motion_history[face_id]
        if face_id in self.last_features:
            del self.last_features[face_id]
        if face_id in self.transition_cooldown:
            del self.transition_cooldown[face_id]
            
    def get_face_stats(self, face_id):
        """
        Yüz ile ilgili mevcut istatistikleri döndürür
        
        Args:
            face_id (str): Yüz kimliği
            
        Returns:
            dict: Yüz ile ilgili istatistikler
        """
        if face_id not in self.mouth_history or not self.mouth_history[face_id]:
            return {"status": "Not enough data"}
            
        last_features = self.last_features.get(face_id, {})
        thresholds = self.person_thresholds.get(face_id, {})
        
        return {
            "speaking": self.speaking_state.get(face_id, False),
            "confidence": self.speaking_confidence.get(face_id, 0),
            "speaking_time": self.speaking_durations.get(face_id, 0),
            "mouth_features": last_features,
            "thresholds": thresholds,
            "history_size": len(self.mouth_history.get(face_id, []))
        }
