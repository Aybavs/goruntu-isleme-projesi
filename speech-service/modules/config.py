"""
Konuşma tespiti servisi için konfigürasyon yönetimi
"""
import os
import logging
from pathlib import Path
import json
try:
    import yaml
except ImportError:
    yaml = None
    logging.getLogger("speech-service").warning("yaml modülü bulunamadı, YAML konfigürasyon dosyaları kullanılamayacak.")

logger = logging.getLogger("speech-service")

class Config:
    """Konfigürasyon yönetim sınıfı"""
    
    def __init__(self, config_path=None):
        """
        Konfigürasyon sınıfını başlatır
        
        Args:
            config_path (str, optional): Konfigürasyon dosyası yolu
        """
        self._config = self._load_default_config()
        self._load_from_file(config_path)
        self._load_from_env()
        
        logger.info("Konfigürasyon yüklendi")
        
    def _load_default_config(self):
        """Varsayılan konfigürasyon değerlerini yükler"""
        return {
            # Servis ayarları
            'host': '0.0.0.0',
            'port': 50053,
            'max_workers': 10,
            
            # Konuşma tespiti parametreleri
            'variation_threshold': 0.03,       # Yüz landmarkları hareketi eşiği
            'confidence_threshold': 0.35,      # Konuşma tespit güven eşiği
            'cooldown_frames': 3,              # Konuşma sonrası bekleme karesi
            'history_length': 20,              # Tarihçe uzunluğu
            'adaptation_rate': 0.08,           # Adaptasyon hızı
            'mouth_area_weight': 0.7,          # Ağız alanı ağırlığı
            'mouth_aspect_ratio_weight': 0.3,  # Ağız en-boy oranı ağırlığı
            
            # Loglama ayarları
            'log_level': 'INFO',
            'log_to_file': True,
            'log_file': 'speech_service.log',
            'log_max_bytes': 10485760,         # 10MB
            'log_backup_count': 5
        }
    
    def _load_from_file(self, config_path):
        """Dosyadan konfigürasyon yükler"""
        if not config_path:
            return
            
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Konfigürasyon dosyası bulunamadı: {config_path}")
            return
            
        try:
            ext = config_file.suffix.lower()
            if ext == '.json':
                with open(config_file, 'r', encoding='utf-8') as f:
                    custom_config = json.load(f)
            elif ext in ['.yml', '.yaml']:
                with open(config_file, 'r', encoding='utf-8') as f:
                    custom_config = yaml.safe_load(f)
            else:
                logger.warning(f"Desteklenmeyen konfigürasyon dosyası formatı: {ext}")
                return
                
            self._config.update(custom_config)
            logger.info(f"Konfigürasyon dosyası yüklendi: {config_path}")
        except Exception as e:
            logger.error(f"Konfigürasyon dosyası yüklenirken hata: {str(e)}", exc_info=True)
    
    def _load_from_env(self):
        """Çevre değişkenlerinden konfigürasyon yükler"""
        for key in self._config:
            env_key = key.upper()
            env_value = os.getenv(env_key)
            
            if not env_value:
                continue
                
            # Değer tipi dönüşümü
            if isinstance(self._config[key], bool):
                self._config[key] = env_value.lower() in ('true', '1', 't', 'y', 'yes')
            elif isinstance(self._config[key], int):
                try:
                    self._config[key] = int(env_value)
                except ValueError:
                    logger.warning(f"Çevre değişkeni {env_key} bir tamsayıya dönüştürülemedi.")
            elif isinstance(self._config[key], float):
                try:
                    self._config[key] = float(env_value)
                except ValueError:
                    logger.warning(f"Çevre değişkeni {env_key} bir ondalık sayıya dönüştürülemedi.")
            else:
                self._config[key] = env_value
    
    def get(self, key, default=None):
        """
        Konfigürasyon değerini getirir
        
        Args:
            key (str): Konfigürasyon anahtarı
            default: Anahtar bulunamazsa dönecek değer
            
        Returns:
            Konfigürasyon değeri
        """
        return self._config.get(key, default)
    
    def get_all(self):
        """
        Tüm konfigürasyon değerlerini getirir
        
        Returns:
            dict: Konfigürasyon değerleri
        """
        return self._config.copy()
        
    def __getitem__(self, key):
        """Sözlük benzeri erişim için"""
        if key not in self._config:
            raise KeyError(f"Konfigürasyon anahtarı bulunamadı: {key}")
        return self._config[key]
        
    def __contains__(self, key):
        """in operatörü desteği"""
        return key in self._config
    
        
    
    
    logger.info("Konfigürasyon ayarları yüklendi")
    
