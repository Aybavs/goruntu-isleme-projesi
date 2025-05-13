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
from concurrent import futures
from collections import defaultdict, deque

# .env dosyasını yükle (varsa)
dotenv_path = Path(__file__).parent / '.env'
if dotenv_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path)

# Dinamik proto klasörü ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'proto'))

# Proto dosyalarını import et
import proto.vision_pb2 as vision_pb2
import proto.vision_pb2_grpc as vision_pb2_grpc

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("speech_service.log")
    ]
)
logger = logging.getLogger("speech-service")


class SpeechDetector:
    """Konuşma tespiti için sınıf"""
    
    def __init__(self, variation_threshold=0.03, confidence_threshold=0.35, cooldown_frames=3):
        """
        SpeechDetector sınıfını başlatır
        """
        # Parametreler için ayarlar
        self.min_variation_threshold = variation_threshold
        self.speaking_confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames
        
        # Kişi bazlı değişkenler 
        self.mouth_history = defaultdict(list)
        self.history_length = 20
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
        self.adaptation_rate = 0.08
        
        logger.info("Konuşma tespit modülü başlatıldı")
        
    def calc_mouth_opening(self, landmarks):
        """
        Verilen landmarks'dan ağız açıklık özelliklerini hesaplar
        """
        # Landmark verisi float listesi olarak gelir, yeniden yapılandır
        if not landmarks or len(landmarks) < 136:  # 68 nokta * 2 (x,y)
            return {
                'normalized_opening': 0.0,
                'horizontal_width': 0.0,
                'avg_lip_distance': 0.0,
                'symmetry': 1.0
            }
            
        # Landmark listesinden noktaları çıkar
        points = []
        for i in range(0, len(landmarks), 2):
            if i+1 < len(landmarks):
                points.append((landmarks[i], landmarks[i+1]))
        
        # Ağız noktaları - indeksler (dlib 68 nokta formatına göre)
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
            'symmetry': symmetry
        }
        
    def detect_speaking(self, face_id, landmarks):
        """
        Konuşma durumunu tespit eder
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
                               self.last_features[face_id]['normalized_opening'])
            
            # Dudak mesafe değişimi
            distance_change = abs(mouth_features['avg_lip_distance'] - 
                                self.last_features[face_id]['avg_lip_distance'])
            
            # Tüm değişimleri birleştir
            total_change = (opening_change * 0.7) + (distance_change * 0.3)
            
            # Değişim geçmişini kaydet
            self.mouth_motion_history[face_id].append(total_change)
        
        # Mevcut değerleri güncelle
        self.last_features[face_id] = mouth_features.copy()

        # Yeterli veri yoksa basit bir kontrol yap
        if len(self.mouth_history[face_id]) < 5:
            # Başlangıç kontrolü
            raw_speaking = mouth_features['normalized_opening'] > 0.3 and mouth_features['avg_lip_distance'] > 3.0
            self.speaking_state[face_id] = raw_speaking
            return raw_speaking

        # Kişiye özel eşik değerini oluştur veya güncelle
        if face_id not in self.person_thresholds or len(self.mouth_history[face_id]) % 10 == 0:
            self._update_person_thresholds(face_id)

        # Mevcut değerler
        current_opening = mouth_features['normalized_opening']
        current_distance = mouth_features['avg_lip_distance']
        
        # Eşik değerleri
        thresholds = self.person_thresholds.get(face_id, {'opening': 0.2, 'distance': 2.0})
        
        # Hareket tespiti
        motion_detected = False
        if len(self.mouth_motion_history[face_id]) >= 4:
            # Son karelerdeki hareket miktarını analiz et
            recent_motion = list(self.mouth_motion_history[face_id])
            
            # Ortalama hareket ve varyans
            mean_motion = np.mean(recent_motion)
            motion_variance = np.var(recent_motion) if len(recent_motion) > 1 else 0
            
            # Hareket tespit kriterleri
            motion_detected = (mean_motion > self.min_variation_threshold) and (motion_variance > 0.0005)

        # Konuşma göstergeleri
        is_open = current_opening > thresholds.get('opening', 0.2)
        has_distance = current_distance > thresholds.get('distance', 2.0)
        
        # Değişim analizi
        recent_frames = min(8, len(self.mouth_history[face_id]))
        recent = self.mouth_history[face_id][-recent_frames:]
        
        # Açıklık değişimi
        openings = [frame['normalized_opening'] for frame in recent]
        opening_std = np.std(openings)
        opening_range = np.max(openings) - np.min(openings)
        
        # Değişim kriteri
        has_variation = (opening_std > self.min_variation_threshold) or (opening_range > 0.15)
        
        # Konuşma skoru hesaplama
        speaking_score = (0.35 * int(is_open) +            # Ağız açıklığı
                         0.20 * int(has_distance) +        # Dudaklar arası mesafe
                         0.25 * int(has_variation) +       # Değişim analizi
                         0.35 * int(motion_detected))      # Hareket tespiti
        
        # Skoru normalize et
        speaking_score = min(1.0, speaking_score)
        
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
        
        return self.speaking_state[face_id]
    
    def _update_person_thresholds(self, face_id):
        """Kişiye özel konuşma eşiklerini günceller"""
        if len(self.mouth_history[face_id]) < 5:
            return
            
        # Geçmiş verilerden istatistikleri hesapla
        openings = [frame['normalized_opening'] for frame in self.mouth_history[face_id]]
        distances = [frame['avg_lip_distance'] for frame in self.mouth_history[face_id]]
        
        # Medyan değerler
        opening_median = np.median(openings)
        distance_median = np.median(distances)
        
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
        
        # Eşik değerlerini güncelle
        self.person_thresholds[face_id] = {
            'opening': max(0.15, opening_thresh),
            'distance': max(1.5, distance_thresh)
        }
        
    def update_speaking_time(self, face_id, is_speaking, current_time):
        """Konuşma süresini günceller"""
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
        """Konuşma süresini döndürür"""
        return self.speaking_durations[face_id]
        
    def clear_face_data(self, face_id):
        """Yüze ait verileri temizler"""
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


class SpeechDetectionServicer(vision_pb2_grpc.SpeechDetectionServiceServicer):
    def __init__(self):
        """Speech Detection Service sınıfını başlatır"""
        self.speech_detector = SpeechDetector(
            variation_threshold=float(os.getenv('VARIATION_THRESHOLD', '0.03')),
            confidence_threshold=float(os.getenv('SPEAKING_CONFIDENCE_THRESHOLD', '0.35')),
            cooldown_frames=int(os.getenv('COOLDOWN_FRAMES', '3'))
        )
        logger.info("Speech Detection Service başlatıldı")
    
    def DetectSpeech(self, request, context):
        """
        Vision Service'den gelen yüz verisini kullanarak konuşma tespiti yapar
        """
        try:
            face_id = request.face_id
            landmarks = list(request.landmarks)
            
            logger.info(f"Konuşma tespiti isteği alındı (Yüz ID: {face_id})")
            
            # Mevcut zaman
            current_time = time.time()
            
            # Konuşma durumunu tespit et
            is_speaking = self.speech_detector.detect_speaking(face_id, landmarks)
            
            # Konuşma süresini güncelle
            self.speech_detector.update_speaking_time(face_id, is_speaking, current_time)
            
            # Konuşma süresini al
            speaking_time = self.speech_detector.get_speaking_time(face_id)
            
            logger.info(f"Yüz ID {face_id} için konuşma durumu: {is_speaking}, süre: {speaking_time:.2f} sn")
            
            # Yanıt oluştur
            return vision_pb2.SpeechResponse(
                is_speaking=is_speaking,
                speaking_time=speaking_time,
                face_id=face_id
            )
            
        except Exception as e:
            logger.error(f"Konuşma tespiti hatası: {str(e)}")
            return vision_pb2.SpeechResponse(
                is_speaking=False,
                speaking_time=0.0,
                face_id=request.face_id
            )


def serve():
    """gRPC sunucusunu başlatır"""
    port = os.getenv('PORT', '50053')
    host = os.getenv('HOST', '0.0.0.0')
    address = f"{host}:{port}"
    max_workers = int(os.getenv('MAX_WORKERS', '10'))
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 10 * 1024 * 1024),  # 10MB
            ('grpc.max_receive_message_length', 10 * 1024 * 1024)  # 10MB
        ]
    )
    
    vision_pb2_grpc.add_SpeechDetectionServiceServicer_to_server(SpeechDetectionServicer(), server)
    server.add_insecure_port(address)
    server.start()
    logger.info(f"SpeechDetectionService gRPC server {address} adresinde çalışıyor...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()