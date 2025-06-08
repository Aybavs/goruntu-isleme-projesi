import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time

logger = logging.getLogger("vision-service")

class FaceTracker:
    """Yüz takibi için sınıf"""
    
    def __init__(self, similarity_threshold=0.4, cleanup_timeout=5.0):
        """
        FaceTracker sınıfını başlatır
        Args:
            similarity_threshold: Yüz eşleşme için benzerlik eşiği
            cleanup_timeout: Yüz temizliği için zaman aşımı (sn)
        """
        self.face_match_threshold = similarity_threshold
        self.face_cleanup_timeout = cleanup_timeout
        
        self.face_database = {}
        self.next_id = 1
        self.last_seen = {}
        
        logger.info("Yüz takip modülü başlatıldı")
        
    def identify_face(self, face_encoding, current_time):
        """
        Yüz özniteliklerine göre mevcut bir ID bulur veya yeni ID atar
        Args:
            face_encoding: Yüz öznitelikleri
            current_time: Mevcut zaman
        Returns:
            Yüz ID'si (int)
        """
        best_match_id = None
        best_match_score = -1

        # Kayıtlı yüzlerle karşılaştır
        for face_id, stored_encoding in self.face_database.items():
            # Benzerlik skoru hesapla (cosine similarity)
            similarity = cosine_similarity([face_encoding], [stored_encoding])[0][0]

            if similarity > self.face_match_threshold and similarity > best_match_score:
                best_match_id = face_id
                best_match_score = similarity

        # Eşleşme bulunamadıysa yeni ID ata
        if best_match_id is None:
            best_match_id = self.next_id
            self.face_database[best_match_id] = face_encoding
            self.next_id += 1
            logger.info(f"Yeni yüz tespit edildi. ID: {best_match_id}")
        else:
            # Kayan ortalama ile öznitelikleri güncelle
            self.face_database[best_match_id] = 0.7 * self.face_database[best_match_id] + 0.3 * face_encoding

        # Son görülme zamanını güncelle
        self.last_seen[best_match_id] = current_time

        return best_match_id
        
    def clean_old_faces(self, current_time, callback=None):
        """
        Belirli bir süre görünmeyen yüzleri temizler
        Args:
            current_time: Mevcut zaman
            callback: Yüz silindiğinde çağrılacak fonksiyon (face_id)
        Returns:
            Silinen yüz ID'lerinin listesi
        """
        ids_to_remove = []
        for face_id, last_time in self.last_seen.items():
            if current_time - last_time > self.face_cleanup_timeout:
                ids_to_remove.append(face_id)

        for face_id in ids_to_remove:
            if face_id in self.face_database:
                del self.face_database[face_id]
            if face_id in self.last_seen:
                del self.last_seen[face_id]
                
            # Callback fonksiyonu varsa çağır
            if callback is not None:
                callback(face_id)
                
            logger.info(f"Yüz {face_id} artık görünmediği için temizlendi")
                
        return ids_to_remove
