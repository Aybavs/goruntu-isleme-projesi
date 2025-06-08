"""
Speaking Time Tracker - Konuşma süresi hesaplama modülü
Her kişi için konuşma sürelerini takip eder ve hesaplar
"""
import time
import logging
from collections import defaultdict
from typing import Dict, Optional

logger = logging.getLogger("speaking-time-tracker")

class SpeakingTimeTracker:
    """Konuşma sürelerini takip eden sınıf"""
    
    def __init__(self, session_timeout: float = 300.0):
        """
        Speaking Time Tracker'ı başlatır
        
        Args:
            session_timeout: Oturum zaman aşımı (saniye) - bu süre sonra veriler temizlenir
        """
        self.session_timeout = session_timeout
        
        # Her face_id için konuşma durumu takibi
        self.speaking_sessions: Dict[int, Dict] = defaultdict(lambda: {
            'is_currently_speaking': False,
            'current_session_start': None,
            'total_speaking_time': 0.0,
            'session_count': 0,
            'last_activity': time.time(),
            'current_session_time': 0.0
        })
        
        logger.info("Speaking Time Tracker başlatıldı")
    
    def update_speaking_status(self, face_id: int, is_speaking: bool) -> float:
        """
        Bir kişinin konuşma durumunu günceller ve toplam konuşma süresini döndürür
        
        Args:
            face_id: Kişi ID'si
            is_speaking: Şu anda konuşuyor mu?
            
        Returns:
            float: Toplam konuşma süresi (saniye)
        """
        current_time = time.time()
        session = self.speaking_sessions[face_id]
        session['last_activity'] = current_time
        
        # Durum değişikliği kontrolü
        was_speaking = session['is_currently_speaking']
        
        if is_speaking and not was_speaking:
            # Konuşma başladı
            session['is_currently_speaking'] = True
            session['current_session_start'] = current_time
            session['session_count'] += 1
            session['current_session_time'] = 0.0
            
            logger.debug(f"Face {face_id}: Konuşma başladı (Oturum #{session['session_count']})")
            
        elif not is_speaking and was_speaking:
            # Konuşma durdu
            if session['current_session_start'] is not None:
                session_duration = current_time - session['current_session_start']
                session['total_speaking_time'] += session_duration
                session['current_session_time'] = 0.0
                
                logger.debug(f"Face {face_id}: Konuşma durdu (Süre: {session_duration:.2f}s, Toplam: {session['total_speaking_time']:.2f}s)")
            
            session['is_currently_speaking'] = False
            session['current_session_start'] = None
            
        elif is_speaking and was_speaking:
            # Konuşma devam ediyor - mevcut oturum süresini güncelle
            if session['current_session_start'] is not None:
                session['current_session_time'] = current_time - session['current_session_start']
        
        return self.get_total_speaking_time(face_id)
    
    def get_total_speaking_time(self, face_id: int) -> float:
        """
        Bir kişinin toplam konuşma süresini döndürür
        
        Args:
            face_id: Kişi ID'si
            
        Returns:
            float: Toplam konuşma süresi (saniye)
        """
        session = self.speaking_sessions[face_id]
        total_time = session['total_speaking_time']
        
        # Eğer şu anda konuşuyorsa, mevcut oturum süresini de ekle
        if session['is_currently_speaking'] and session['current_session_start'] is not None:
            current_session_duration = time.time() - session['current_session_start']
            total_time += current_session_duration
        
        return total_time
    
    def get_current_session_time(self, face_id: int) -> float:
        """
        Mevcut konuşma oturumunun süresini döndürür
        
        Args:
            face_id: Kişi ID'si
            
        Returns:
            float: Mevcut oturum süresi (saniye)
        """
        session = self.speaking_sessions[face_id]
        
        if session['is_currently_speaking'] and session['current_session_start'] is not None:
            return time.time() - session['current_session_start']
        
        return 0.0
    
    def get_speaking_stats(self, face_id: int) -> Dict:
        """
        Bir kişinin detaylı konuşma istatistiklerini döndürür
        
        Args:
            face_id: Kişi ID'si
            
        Returns:
            dict: Konuşma istatistikleri
        """
        session = self.speaking_sessions[face_id]
        current_time = time.time()
        
        stats = {
            'face_id': face_id,
            'is_currently_speaking': session['is_currently_speaking'],
            'total_speaking_time': self.get_total_speaking_time(face_id),
            'current_session_time': self.get_current_session_time(face_id),
            'session_count': session['session_count'],
            'last_activity': session['last_activity'],
            'time_since_last_activity': current_time - session['last_activity']
        }
        
        return stats
    
    def is_currently_speaking(self, face_id: int) -> bool:
        """
        Kişinin şu anda konuşup konuşmadığını döndürür
        
        Args:
            face_id: Kişi ID'si
            
        Returns:
            bool: Şu anda konuşuyor mu?
        """
        return self.speaking_sessions[face_id]['is_currently_speaking']
    
    def clear_face_data(self, face_id: int):
        """
        Belirli bir kişinin verilerini temizler
        
        Args:
            face_id: Temizlenecek kişi ID'si
        """
        if face_id in self.speaking_sessions:
            stats = self.get_speaking_stats(face_id)
            logger.info(f"Face {face_id} verileri temizleniyor - Son istatistikler: {stats}")
            del self.speaking_sessions[face_id]
    
    def cleanup_inactive_faces(self) -> int:
        """
        Uzun süre aktif olmayan yüzlerin verilerini temizler
        
        Returns:
            int: Temizlenen yüz sayısı
        """
        current_time = time.time()
        faces_to_remove = []
        
        for face_id, session in self.speaking_sessions.items():
            time_since_activity = current_time - session['last_activity']
            
            if time_since_activity > self.session_timeout:
                faces_to_remove.append(face_id)
        
        for face_id in faces_to_remove:
            self.clear_face_data(face_id)
        
        if faces_to_remove:
            logger.info(f"{len(faces_to_remove)} inaktif yüz temizlendi: {faces_to_remove}")
        
        return len(faces_to_remove)
    
    def get_all_stats(self) -> Dict:
        """
        Tüm yüzlerin istatistiklerini döndürür
        
        Returns:
            dict: Tüm yüzlerin istatistikleri
        """
        all_stats = {}
        for face_id in self.speaking_sessions.keys():
            all_stats[face_id] = self.get_speaking_stats(face_id)
        
        return all_stats
    
    def reset_all_data(self):
        """
        Tüm konuşma verilerini sıfırlar
        """
        face_count = len(self.speaking_sessions)
        self.speaking_sessions.clear()
        logger.info(f"Tüm konuşma verileri sıfırlandı ({face_count} yüz)")
