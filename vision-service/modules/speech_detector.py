import numpy as np
from collections import defaultdict, deque
import logging
import time

logger = logging.getLogger("vision-service")

class SpeechDetector:
    """Konuşma tespiti için sınıf"""
    
    def __init__(self, variation_threshold=0.03, confidence_threshold=0.35, cooldown_frames=3):
        """
        SpeechDetector sınıfını başlatır - geliştirilmiş parametreler
        Args:
            variation_threshold: Konuşma tespitinde minimum değişim eşiği
            confidence_threshold: Konuşma güven skoru eşiği
            cooldown_frames: Durumlar arası geçiş için bekleme karesi sayısı
        """
        # Parametreler için daha hassas ayarlar
        self.min_variation_threshold = variation_threshold
        self.speaking_confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames
        
        # Kişi bazlı değişkenler 
        self.mouth_history = defaultdict(list)
        self.history_length = 20  # Daha uzun geçmiş - daha stabil sonuçlar
        self.person_thresholds = {}
        self.speaking_confidence = defaultdict(float)
        self.speaking_state = defaultdict(bool)
        self.transition_cooldown = defaultdict(int)
        
        # Konuşma süresi takibi için değişkenler
        self.speaking_start_times = {}
        self.speaking_durations = defaultdict(float)
        
        # Geliştirilmiş konuşma tespiti için yeni değişkenler
        self.mouth_motion_history = defaultdict(lambda: deque(maxlen=12))  # Uzatılmış hareket geçmişi
        self.speech_patterns = defaultdict(list)  # Konuşma paternlerini öğrenmek için
        self.person_speaking_style = {}  # Kişisel konuşma stili (bazı insanlar ağzını çok açar)
        self.last_features = defaultdict(dict)  # Son kare özellikleri
        
        # Kişisel konuşma stiline adapte olmak için öğrenme oranı
        self.adaptation_rate = 0.08  # Daha yavaş adapte ol (0.1 → 0.08)
        
        # Konuşma modelleme parametreleri
        self.rhythm_analysis_window = 15  # Ritim analizi penceresi
        self.min_cycles_for_speech = 1.2  # Konuşma için gereken minimum çevrim sayısı (1.0 → 1.2)
        self.cycle_threshold = 0.12  # Bir çevrim için gereken minimum değişim (0.15 → 0.12)
        
        # Ekstra filtreler
        self.min_speech_duration = 0.3  # En az bu kadar süren konuşmalar geçerli kabul edilir (saniye)
        self.max_silence_gap = 0.5  # Konuşmada bu süre sessizlik olursa aynı konuşma kabul et (saniye)
        self.face_detection_confidence = 0.95  # Yüz tespitinde minimum güven
        
        logger.info("Geliştirilmiş konuşma tespit modülü başlatıldı")
        
    def calc_mouth_opening(self, landmarks):
        """
        Ağız açıklığını ve diğer dudak özelliklerini gelişmiş şekilde hesaplar
        Args:
            landmarks: dlib yüz landmark noktaları
        Returns:
            Ağız özellikleri sözlüğü
        """
        # Landmark noktaları yoksa boş özellikleri döndür
        if landmarks is None:
            return {
                'vertical_opening': 0.0,
                'horizontal_width': 0.0,
                'opening_ratio': 0.0,
                'normalized_opening': 0.0,
                'mouth_area': 0.0,
                'normalized_area': 0.0,
                'avg_lip_distance': 0.0,
                'max_lip_distance': 0.0,
                'lip_curl': 0.0,
                'symmetry': 1.0
            }
            
        # Ağız noktaları - daha fazla nokta kullan
        top_lip_indices = [61, 62, 63, 64, 65]
        bottom_lip_indices = [67, 66, 65, 64, 63]
        
        # Köşe noktaları (konuşmada önemli)
        left_corner = 48
        right_corner = 54
        
        # Orta noktalar
        top_middle = 62
        bottom_middle = 66

        # Dikey açıklık - orta noktalarda
        vertical_opening = abs(landmarks.part(top_middle).y - landmarks.part(bottom_middle).y)
        
        # Yatay genişlik
        horizontal_width = abs(landmarks.part(left_corner).x - landmarks.part(right_corner).x)
        
        # Dudak kıvrımı - konuşma/gülümseme/kızgınlık analizi için önemli
        left_top = landmarks.part(left_corner).y
        right_top = landmarks.part(right_corner).y
        middle_top = landmarks.part(top_middle).y
        lip_curl = ((left_top + right_top) / 2) - middle_top
        
        # Simetri - dudağın simetrik olup olmadığı
        left_side_opening = abs(landmarks.part(61).y - landmarks.part(67).y)  # Sol taraf
        right_side_opening = abs(landmarks.part(65).y - landmarks.part(63).y)  # Sağ taraf
        
        # 1'e yakın değer daha simetrik
        symmetry = min(left_side_opening, right_side_opening) / max(left_side_opening, right_side_opening) if max(left_side_opening, right_side_opening) > 0 else 1.0
        
        # Ağız alanı - daha doğru ölçüm
        top_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in top_lip_indices]
        bottom_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in bottom_lip_indices]
        
        # Çokgen alanı hesapla
        mouth_area = 0
        for i in range(len(top_points)-1):
            mouth_area += abs((top_points[i][0] * bottom_points[i+1][1] - bottom_points[i+1][0] * top_points[i][1]) -
                             (top_points[i+1][0] * bottom_points[i][1] - bottom_points[i][0] * top_points[i+1][1]))

        # Normalize edilmiş değerler
        normalized_opening = vertical_opening / (horizontal_width * 0.5) if horizontal_width > 0 else 0
        normalized_area = mouth_area / (horizontal_width * vertical_opening) if horizontal_width * vertical_opening > 0 else 0

        # Üst ve alt dudak konturları arasındaki mesafe (çoklu noktalar)
        distances = []
        for i in range(len(top_lip_indices)):
            top_pt = (landmarks.part(top_lip_indices[i]).x, landmarks.part(top_lip_indices[i]).y)
            bottom_pt = (landmarks.part(bottom_lip_indices[i]).x, landmarks.part(bottom_lip_indices[i]).y)
            distance = np.sqrt((top_pt[0] - bottom_pt[0])**2 + (top_pt[1] - bottom_pt[1])**2)
            distances.append(distance)
        
        avg_lip_distance = np.mean(distances) if distances else 0
        max_lip_distance = np.max(distances) if distances else 0

        # Dudak şekli özellikleri - genişletilmiş set
        mouth_features = {
            'vertical_opening': vertical_opening,
            'horizontal_width': horizontal_width,
            'opening_ratio': vertical_opening / horizontal_width if horizontal_width > 0 else 0,
            'normalized_opening': normalized_opening,
            'mouth_area': mouth_area,
            'normalized_area': normalized_area,
            'avg_lip_distance': avg_lip_distance,
            'max_lip_distance': max_lip_distance,
            'lip_curl': lip_curl,
            'symmetry': symmetry
        }

        return mouth_features
        
    def detect_speaking(self, face_id, mouth_features):
        """
        Geliştirilmiş konuşma algılama - daha stabil ve hassas
        Args:
            face_id: Kişi ID'si
            mouth_features: Ağız özellikleri sözlüğü
        Returns:
            Konuşma durumu (boolean)
        """
        # Ağız özelliklerini geçmişe ekle
        self.mouth_history[face_id].append(mouth_features)

        # Geçmişi belirli bir boyutta sınırla
        if len(self.mouth_history[face_id]) > self.history_length:
            self.mouth_history[face_id].pop(0)
            
        # Son kareden beri olan değişimi kaydet (hareket tespiti için)
        if face_id in self.last_features:
            # Ağız açıklığındaki değişim
            opening_change = abs(mouth_features['normalized_opening'] - 
                               self.last_features[face_id]['normalized_opening'])
            
            # Alan değişimi 
            area_change = abs(mouth_features['normalized_area'] - 
                           self.last_features[face_id]['normalized_area'])
            
            # Dudak kıvrım değişimi
            curl_change = abs(mouth_features['lip_curl'] - 
                           self.last_features[face_id]['lip_curl'])
            
            # Dudak mesafe değişimi (ek)
            distance_change = abs(mouth_features['avg_lip_distance'] - 
                                self.last_features[face_id]['avg_lip_distance'])
            
            # Tüm değişimleri birleştir (ağırlıklı - yeni ağırlıklar)
            total_change = (opening_change * 0.4) + (area_change * 0.25) + (curl_change * 0.15) + (distance_change * 0.2)
            
            # Değişim geçmişini kaydet
            self.mouth_motion_history[face_id].append(total_change)
        
        # Mevcut değerleri güncelle
        self.last_features[face_id] = mouth_features.copy()

        # Yeterli veri yoksa basit bir kontrol yap
        if len(self.mouth_history[face_id]) < 5:
            # Daha temkinli başlangıç kontrolü
            raw_speaking = mouth_features['normalized_opening'] > 0.3 and mouth_features['avg_lip_distance'] > 3.0
            self.speaking_state[face_id] = raw_speaking
            return raw_speaking

        # Ritim analizi - konuşma için tipik ağız/dudak hareket ritmini tespit et
        rhythm_detected = self._detect_speech_rhythm(face_id)

        # Kişiye özel eşik değerini oluştur veya güncelle (her 10 karede bir)
        if face_id not in self.person_thresholds or len(self.mouth_history[face_id]) % 10 == 0:
            self._update_person_thresholds(face_id)

        # Mevcut değerler
        current_opening = mouth_features['normalized_opening']
        current_area = mouth_features['normalized_area']
        current_distance = mouth_features['avg_lip_distance']
        current_symmetry = mouth_features['symmetry']
        
        # Eşik değerleri
        thresholds = self.person_thresholds[face_id]
        
        # Hareket tespiti - son 8 karedeki hareket analizi
        motion_detected = False
        if len(self.mouth_motion_history[face_id]) >= 4:  # En az 4 kare gerekli
            # Son birkaç karedeki hareket miktarını analiz et
            recent_motion = list(self.mouth_motion_history[face_id])
            
            # Ortalama hareket
            mean_motion = np.mean(recent_motion)
            
            # Hareket değişkenliği - konuşma daha değişken harekete sahiptir
            motion_variance = np.var(recent_motion) if len(recent_motion) > 1 else 0
            
            # Hem ortalama hareket hem de değişkenlik değerini kontrol et
            motion_detected = (mean_motion > self.min_variation_threshold * 1.2) and (motion_variance > 0.0005)

        # Konuşma göstergeleri - daha fazla özellik kullan
        is_open = current_opening > thresholds.get('opening', 0.2)
        has_area = current_area > thresholds.get('area', 0.1)
        has_distance = current_distance > thresholds.get('distance', 2.0)
        is_symmetric = current_symmetry > 0.7  # Konuşmada dudak genelde simetriktir
        
        # Değişim analizi (standart sapma ve değişim aralığı)
        recent_frames = min(8, len(self.mouth_history[face_id]))
        recent = self.mouth_history[face_id][-recent_frames:]
        
        # Farklı özelliklerdeki değişimler
        openings = [frame['normalized_opening'] for frame in recent]
        areas = [frame['normalized_area'] for frame in recent]
        distances = [frame['avg_lip_distance'] for frame in recent]
        
        # İstatistiksel analizler
        opening_std = np.std(openings)
        opening_range = np.max(openings) - np.min(openings)
        
        # Zigzag hareketi tespit etme - konuşmada değişim sık yön değiştirir
        zigzag_score = 0
        if len(openings) >= 4:
            direction_changes = 0
            for i in range(2, len(openings)):
                prev_dir = openings[i-1] - openings[i-2]
                curr_dir = openings[i] - openings[i-1]
                if (prev_dir * curr_dir) < 0:  # Yön değişimi
                    direction_changes += 1
            zigzag_score = direction_changes / (len(openings) - 2)  # Normalize edilmiş

        # Varyasyon değeri (birden fazla özellik üzerinden)
        has_variation = ((opening_std > self.min_variation_threshold * 1.2) or 
                        (opening_range > 0.15) or 
                        (zigzag_score > 0.4))  # Zigzag kriteri eklendi
        
        # Ritim faktörü
        has_rhythm = rhythm_detected
        
        # Ağırlıklı kombine skor (yeni ağırlıklar)
        speaking_score = (0.35 * int(is_open) +            # Ağız açıklığı
                         0.15 * int(has_area) +            # Ağız alanı
                         0.15 * int(has_distance) +        # Dudaklar arası mesafe
                         0.30 * int(has_variation) +       # Değişim analizi (arttırıldı)
                         0.35 * int(motion_detected) +     # Hareket tespiti (arttırıldı)
                         0.25 * int(has_rhythm) +          # Ritim tespiti (arttırıldı)
                         0.1 * int(is_symmetric))          # Simetri kontrolü
        
        # Skoru normalize et (1.0'dan büyük olabilir)
        speaking_score = min(1.0, speaking_score)
        
        # Konuşma güven skorunu güncelle (daha hassas)
        if speaking_score > 0.6:  # Eşik değeri arttırıldı
            # Konuşma göstergeleri varsa güven skorunu arttır (daha yavaş artış)
            self.speaking_confidence[face_id] = min(1.0, self.speaking_confidence[face_id] + 0.12)
        elif speaking_score < 0.25:  # Eşik değeri düşürüldü
            # Yoksa düşür (daha hızlı düşme)
            self.speaking_confidence[face_id] = max(0.0, self.speaking_confidence[face_id] - 0.15)
        else:
            # Ara durumlarda mevcut durumu koru (stabilite için)
            pass
        
        # Mevcut karara karar ver
        raw_speaking_state = self.speaking_confidence[face_id] > self.speaking_confidence_threshold
        
        # Soğuma süreci - daha uzun soğuma süresi (stabilite için)
        cooldown_value = self.cooldown_frames + 1  # Bir kare daha ekle
        if raw_speaking_state != self.speaking_state[face_id]:
            if self.transition_cooldown[face_id] >= cooldown_value:
                self.speaking_state[face_id] = raw_speaking_state
                self.transition_cooldown[face_id] = 0
            else:
                self.transition_cooldown[face_id] += 1
        else:
            self.transition_cooldown[face_id] = 0
        
        return self.speaking_state[face_id]
    
    def _update_person_thresholds(self, face_id):
        """
        Geliştirilmiş kişiye özel konuşma eşikleri - daha iyi adaptasyon
        Args:
            face_id: Kişi ID'si
        """
        if len(self.mouth_history[face_id]) < 5:
            return
            
        # Geçmiş verilerden istatistikleri hesapla
        openings = [frame['normalized_opening'] for frame in self.mouth_history[face_id]]
        areas = [frame['normalized_area'] for frame in self.mouth_history[face_id]]
        distances = [frame['avg_lip_distance'] for frame in self.mouth_history[face_id]]
        lip_curls = [frame.get('lip_curl', 0) for frame in self.mouth_history[face_id]]
        
        # Robust istatistikler (aykırı değerlerden daha az etkilenir)
        # Medyanlar
        opening_median = np.median(openings)
        area_median = np.median(areas)
        distance_median = np.median(distances)
        
        # Çeyrekler arası aralık (IQR) - aykırı değerlere karşı daha sağlam
        opening_q75 = np.percentile(openings, 75)
        opening_q25 = np.percentile(openings, 25)
        opening_iqr = opening_q75 - opening_q25
        
        # Dinlenme halindeki ağız açıklığı (alt persentiller)
        resting_opening = np.percentile(openings, 30)  # Dinlenme durumu genelde daha düşük değerlere sahip
        
        # Dinamik eşikler (kişisel)
        # Dinlenme + fark - konuşma için minimum eşik
        opening_thresh = resting_opening + (opening_iqr * 0.7)
        
        # Alan eşikleri (medyan bazlı)
        area_thresh = area_median * 1.2  # %20 daha fazla alan
        
        # Mesafe eşiği (medyan bazlı)
        distance_thresh = distance_median * 1.15  # %15 daha fazla mesafe
        
        # Kişisel konuşma stilini öğren ve güncelle
        if face_id in self.person_speaking_style:
            # Mevcut tarzı kademeli olarak güncelle
            old_style = self.person_speaking_style[face_id]
            update_rate = self.adaptation_rate
            
            # Yeni değerleri mevcut stile entegre et
            updated_style = {
                'median_opening': old_style['median_opening'] * (1-update_rate) + opening_median * update_rate,
                'opening_variation': old_style.get('opening_variation', 0) * (1-update_rate) + opening_iqr * update_rate,
                'resting_opening': old_style.get('resting_opening', 0) * (1-update_rate) + resting_opening * update_rate,
                'speaking_intensity': old_style.get('speaking_intensity', 1.0)  # Konuşma yoğunluğu faktörü
            }
            
            # Konuşma tarzı faktörünü güncelle
            if hasattr(self, 'speaking_state') and face_id in self.speaking_state:
                # Aktif konuşma durumlarından öğren
                if self.speaking_state[face_id]:
                    # Konuşma sırasında yüksek açıklık değerleri
                    active_openings = [o for i, o in enumerate(openings[-10:]) 
                                      if i < len(self.speaking_state) and self.speaking_state[face_id]]
                    if active_openings:
                        # Konuşma yoğunluğu faktörünü güncelle
                        speaking_intensity = np.mean(active_openings) / (opening_median if opening_median > 0 else 1)
                        # Sınırla ve güncelle
                        speaking_intensity = max(0.8, min(2.0, speaking_intensity))
                        updated_style['speaking_intensity'] = old_style.get('speaking_intensity', 1.0) * 0.9 + speaking_intensity * 0.1
            
            self.person_speaking_style[face_id] = updated_style
        else:
            # İlk kez için stil oluştur
            self.person_speaking_style[face_id] = {
                'median_opening': opening_median,
                'opening_variation': opening_iqr,
                'resting_opening': resting_opening,
                'speaking_intensity': 1.0  # Başlangıç değeri
            }
        
        # Kişisel konuşma stiline göre eşikleri ayarla
        style = self.person_speaking_style[face_id]
        
        # Konuşma yoğunluğu faktörü (1.0 normal, daha küçük sessiz konuşanlar için, daha büyük ağzını çok açan konuşmacılar için)
        intensity_factor = 1.0 / style.get('speaking_intensity', 1.0)
        
        # Kombine eşik değeri - kişisel stil faktörleriyle ayarlanmış
        self.person_thresholds[face_id] = {
            'opening': max(0.15, opening_thresh * intensity_factor),
            'area': max(0.1, area_thresh * intensity_factor),
            'distance': max(1.5, distance_thresh),
            'lip_curl': np.percentile(lip_curls, 60)  # Dudak kıvrım eşiği
        }
        
        logger.debug(f"ID {face_id} için kişiselleştirilmiş konuşma eşikleri: açıklık={self.person_thresholds[face_id]['opening']:.2f}, yoğunluk={style.get('speaking_intensity', 1.0):.2f}")
    
    def _detect_speech_rhythm(self, face_id):
        """
        Geliştirilmiş konuşma ritmi tespiti - daha doğru algılama
        Args:
            face_id: Kişi ID'si
        Returns:
            Boolean: Ritim tespit edildi mi
        """
        if len(self.mouth_history[face_id]) < self.rhythm_analysis_window:
            return False
            
        # Son N karedeki ağız açıklık değişimini analiz et
        recent_frames = self.mouth_history[face_id][-self.rhythm_analysis_window:]
        
        # Birden fazla özelliği analiz et
        openings = [frame['normalized_opening'] for frame in recent_frames]
        distances = [frame['avg_lip_distance'] for frame in recent_frames]
        
        # Sinyal işleme: hareketli ortalama filtresi ile gürültüyü azalt
        window_size = 3
        if len(openings) >= window_size:
            smoothed_openings = []
            for i in range(len(openings) - window_size + 1):
                smoothed_openings.append(sum(openings[i:i+window_size]) / window_size)
        else:
            smoothed_openings = openings
        
        # Açılma/kapanma çevrimlerini say (geliştirilmiş yöntem)
        cycles = 0
        increasing = None
        peak_detected = False
        valley_detected = False
        
        threshold_ratio = 0.12  # Değişim eşiği - daha hassas
        
        # Yerel maksimum/minimum noktalarını bul
        local_peaks = []
        local_valleys = []
        
        for i in range(1, len(smoothed_openings)-1):
            # Yerel maksimum
            if smoothed_openings[i] > smoothed_openings[i-1] and smoothed_openings[i] > smoothed_openings[i+1]:
                local_peaks.append(i)
            
            # Yerel minimum
            if smoothed_openings[i] < smoothed_openings[i-1] and smoothed_openings[i] < smoothed_openings[i+1]:
                local_valleys.append(i)
        
        # Anlamlı yerel extremumleri filtrele (değişim büyüklüğüne göre)
        significant_cycles = 0
        max_val = max(smoothed_openings) if smoothed_openings else 0
        min_val = min(smoothed_openings) if smoothed_openings else 0
        range_val = max_val - min_val
        
        for i in range(min(len(local_peaks), len(local_valleys))):
            peak = local_peaks[i]
            valley = local_valleys[i]
            
            if abs(smoothed_openings[peak] - smoothed_openings[valley]) > (range_val * threshold_ratio):
                significant_cycles += 0.5
        
        # Alternatif yöntem: Varyasyon sıklığı - konuşma için tipik değişim frekansı
        if len(smoothed_openings) > 3:
            zero_crossings = 0
            mean_val = np.mean(smoothed_openings)
            
            # Ortalamadan sapmaların işareti değiştiğinde sayaç artır (konuşmada sık görülür)
            for i in range(1, len(smoothed_openings)):
                if (smoothed_openings[i-1] - mean_val) * (smoothed_openings[i] - mean_val) < 0:
                    zero_crossings += 1
            
            # Normalize edilmiş değişim frekansı
            variation_frequency = zero_crossings / (len(smoothed_openings) - 1)
            
            # Konuşma için tipik değişim frekansı (deneysel değer)
            typical_speech_freq = variation_frequency > 0.25 and variation_frequency < 0.7
        else:
            typical_speech_freq = False
        
        # Kombinasyon: Hem çevrim hem de değişim frekansı
        rhythm_score = significant_cycles * 0.7 + int(typical_speech_freq) * 0.6
        
        # Frekans analizi: FFT kullanarak periyodiklik kontrolü
        has_periodicity = False
        if len(openings) >= 8:
            try:
                # NumPy FFT kullanarak frekans analizi
                fft_data = np.abs(np.fft.rfft(openings - np.mean(openings)))
                # DC bileşenini çıkar (ilk eleman)
                fft_data = fft_data[1:] if len(fft_data) > 1 else fft_data
                
                # Güçlü frekans bileşeninin varlığını kontrol et (konuşmada periyodiklik olur)
                if len(fft_data) > 0:
                    max_amplitude = np.max(fft_data)
                    total_power = np.sum(fft_data)
                    
                    # Dominant frekans gücü yeterince yüksek mi?
                    dominance_ratio = max_amplitude / total_power if total_power > 0 else 0
                    has_periodicity = dominance_ratio > 0.3  # Konuşma için tipik değer
            except:
                # FFT başarısız olursa (nadiren olabilir)
                pass
        
        # Final sonuç - birden fazla yöntemin kombinasyonu
        rhythm_detected = (significant_cycles >= self.min_cycles_for_speech or 
                          typical_speech_freq or 
                          (rhythm_score > 0.8) or 
                          has_periodicity)
        
        return rhythm_detected
        
    def update_speaking_time(self, face_id, is_speaking, current_time):
        """
        Kişinin konuşma süresini hesaplar
        Args:
            face_id: Kişi ID'si
            is_speaking: Konuşma durumu
            current_time: Mevcut zaman
        """
        if is_speaking:
            # Konuşmaya başladıysa başlangıç zamanını kaydet
            if face_id not in self.speaking_start_times:
                self.speaking_start_times[face_id] = current_time
        else:
            # Konuşmayı bitirdiyse süreyi hesapla ve ekle
            if face_id in self.speaking_start_times:
                elapsed = current_time - self.speaking_start_times[face_id]
                # Çok kısa süreli konuşmaları filtrele (0.5 saniyeden kısa)
                if elapsed > 0.5:  
                    self.speaking_durations[face_id] += elapsed
                del self.speaking_start_times[face_id]
                
    def get_speaking_time(self, face_id):
        """
        Kişinin konuşma süresini döndürür
        Args:
            face_id: Kişi ID'si
        Returns:
            Toplam konuşma süresi (saniye)
        """
        return self.speaking_durations[face_id]
        
    def clear_face_data(self, face_id):
        """
        Yüze ait verileri temizler
        Args:
            face_id: Kişi ID'si
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
