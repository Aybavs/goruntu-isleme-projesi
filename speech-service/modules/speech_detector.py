"""
Çok daha hassas konuşma tespiti modülü
Düşük eşikler ve hızlı tepki için optimize edildi.
"""
import numpy as np
from collections import defaultdict, deque
import logging

logger = logging.getLogger("speech-detector")

class SpeechDetector:
    def __init__(
        self,
        base_threshold: float = 0.08,           # daha düşük eşik
        history_size: int = 8,
        cooldown_frames: int = 1,               # tek karede geçiş
        max_silence_frames: int = 10,
        calibration_frames: int = 10,           # hızlı kalibrasyon
        alpha: float = 1.0,                     # dinamik eşik standart sapma çarpanı
        blend_factor: float = 0.8,              # %80 dinamik eşiğe ağırlık
        min_opening_threshold: float = 0.1,     # %10 oran eşik
        smooth_window: int = 1,                 # anlık tepki için pencere=1
    ):
        self.base_threshold = base_threshold
        self.history_size = history_size
        self.cooldown_frames = cooldown_frames
        self.max_silence_frames = max_silence_frames

        # Kalibrasyon
        self.calibration_frames = calibration_frames
        self.alpha = alpha
        self.calib_counts = defaultdict(int)
        self.calib_history = defaultdict(list)
        self.dynamic_threshold = {}

        # Eşik karışımı
        self.blend_factor = blend_factor
        self.min_opening_threshold = min_opening_threshold

        # Hareket filtresi
        self.smooth_window = smooth_window
        self.movement_history = defaultdict(lambda: deque(maxlen=smooth_window))

        # Kişi bazlı veri
        self.mouth_history = defaultdict(lambda: deque(maxlen=history_size))
        self.speaking_states = defaultdict(bool)
        self.cooldown_counters = defaultdict(int)
        self.silence_counters = defaultdict(int)

        logger.info("SpeechDetector (çok hassas) başlatıldı")

    def _extract_mouth_features(self, landmarks):
        if not landmarks or len(landmarks) < 136:
            return None
        pts = np.array([[landmarks[i*2], landmarks[i*2+1]] for i in range(48,68)])
        w = np.linalg.norm(pts[6] - pts[0])
        top = np.mean(pts[13:16], axis=0)
        bot = np.mean(pts[16:20], axis=0)
        h = np.linalg.norm(top - bot)
        return {'width': w, 'height': h, 'ratio': h/(w+1e-6)}

    def detect_speaking(self, face_id, landmarks):
        feats = self._extract_mouth_features(landmarks)
        if feats is None:
            return False

        # 1) Kalibrasyon
        if self.calib_counts[face_id] < self.calibration_frames:
            self.calib_history[face_id].append(feats['ratio'])
            self.calib_counts[face_id] += 1
            if self.calib_counts[face_id] == self.calibration_frames:
                arr = np.array(self.calib_history[face_id])
                mu, sigma = arr.mean(), arr.std()
                self.dynamic_threshold[face_id] = mu + self.alpha * sigma
                logger.info(f"Face{face_id} dinamik eşik: {self.dynamic_threshold[face_id]:.4f}")
            return False

        # 2) Geçmişe ekle
        self.mouth_history[face_id].append(feats)
        if len(self.mouth_history[face_id]) < 3:
            return False

        # 3) Hareket analizi
        moving = self._analyze_movement(face_id)

        # 4) Durumu güncelle
        self._update_state(face_id, moving)
        return self.speaking_states[face_id]

    def _analyze_movement(self, face_id):
        recent = list(self.mouth_history[face_id])[-5:]
        heights = [f['height'] for f in recent]
        vars_h = np.var(heights) if len(heights)>1 else 0
        ratios = [f['ratio'] for f in recent]
        vars_r = np.var(ratios) if len(ratios)>1 else 0
        widths = [f['width'] for f in recent]
        vars_w = np.var(widths) if len(widths)>1 else 0

        raw = vars_h + vars_r + 0.5*vars_w
        # Anlık pencere → smooth_window=1
        mh = self.movement_history[face_id]
        mh.append(raw)
        smooth = sum(mh)/len(mh)

        # Eşik karışımı
        dyn = self.dynamic_threshold.get(face_id, self.base_threshold)
        thr = self.base_threshold*(1-self.blend_factor) + dyn*self.blend_factor

        # Çoklu koşullar
        cond1 = vars_h > thr*0.3
        cond2 = vars_r > thr*0.3
        cond3 = vars_w > thr*0.1
        primary = ((cond1 and cond2) or cond3) and (smooth > thr)

        # Düşük açılma eşiği
        opening_ok = np.mean(ratios) > self.min_opening_threshold

        moving = primary and opening_ok
        logger.debug(f"Face{face_id}: smooth={smooth:.4f}, thr={thr:.4f}, moving={moving}")
        return moving

    def _update_state(self, face_id, move):
        curr = self.speaking_states[face_id]
        if move:
            self.silence_counters[face_id] = 0
            if not curr:
                self.cooldown_counters[face_id] += 1
                if self.cooldown_counters[face_id] >= self.cooldown_frames:
                    self.speaking_states[face_id] = True
                    self.cooldown_counters[face_id] = 0
                    logger.info(f"Face{face_id} konuşma başladı")
            else:
                self.cooldown_counters[face_id] = 0
        else:
            self.silence_counters[face_id] += 1
            if curr:
                self.cooldown_counters[face_id] += 1
                if (self.cooldown_counters[face_id] >= self.cooldown_frames or
                    self.silence_counters[face_id] >= self.max_silence_frames):
                    self.speaking_states[face_id] = False
                    self.cooldown_counters[face_id] = 0
                    self.silence_counters[face_id] = 0
                    logger.info(f"Face{face_id} konuşma durdu")
            else:
                self.cooldown_counters[face_id] = 0

    def is_speaking(self, face_id):
        return self.speaking_states.get(face_id, False)

    def get_stats(self, face_id):
        history = self.mouth_history.get(face_id, [])
        thr = self.dynamic_threshold.get(face_id, self.base_threshold)
        return {
            'speaking': self.speaking_states.get(face_id, False),
            'frame_count': len(history),
            'cooldown': self.cooldown_counters.get(face_id, 0),
            'silence': self.silence_counters.get(face_id, 0),
            'threshold': thr
        }

    def clear_all(self):
        for d in (
            self.mouth_history, self.speaking_states, self.cooldown_counters,
            self.silence_counters, self.movement_history,
            self.calib_history, self.calib_counts, self.dynamic_threshold
        ):
            d.clear()
        logger.info("Tüm veriler temizlendi")
