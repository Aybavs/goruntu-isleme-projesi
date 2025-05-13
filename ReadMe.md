# Görüntü İşleme Analiz Sistemi

Bu proje, gerçek zamanlı video akışı üzerinde yüz tespiti, takibi, duygu analizi ve konuşma tespiti gerçekleştiren bir mikroservis mimarisidir. Sistem, dört temel bileşenden oluşmaktadır: Frontend, Gateway, Vision Service, Emotion Service ve Speech Detection Service.

## Mimari Genel Bakış

Sistem aşağıdaki bileşenlerden oluşmaktadır:

- **Frontend**: React ve TypeScript ile yazılmış web arayüzü
- **Gateway**: Node.js tabanlı WebSocket-gRPC köprüsü
- **Vision Service**: Python tabanlı görüntü işleme ve yüz tespiti servisi
- **Emotion Service**: Yüz görüntülerinden duygu analizi yapan servis
- **Speech Detection Service**: Yüz landmark'larına bakarak konuşma tespiti yapan servis

```
Frontend <--WebSocket--> Gateway <--gRPC--> Vision Service
                                 |
                                 +---gRPC---> Emotion Service
                                 |
                                 +---gRPC---> Speech Detection Service
```

## Bileşenler

### Frontend (React/TypeScript)

Frontend uygulaması, kullanıcının kamerasından görüntüleri yakalayan ve bunları WebSocket üzerinden Gateway'e ileten bir web arayüzü sağlar.

**Temel özellikler:**

- Kamera akışı görüntüleme
- WebSocket üzerinden görüntü karelerini Gateway'e gönderme
- Vision Service'ten gelen analiz sonuçlarını görselleştirme:
  - Tespit edilen yüzler
  - Konuşma durumu
  - Duygu analizi sonuçları
  - Konuşma süreleri

**Nasıl çalışır:**

1. Frontend, kamera görüntülerini periyodik olarak yakalar
2. Yakalanan görüntüler WebSocket bağlantısı üzerinden Gateway'e gönderilir
3. Gateway'den dönen yanıtlar işlenir ve arayüzde görselleştirilir
4. Her tespit edilen kişi için bir kart oluşturulur

### Gateway (Node.js)

Gateway, Frontend ile diğer servisler arasında aracılık yapar. WebSocket protokolünü gRPC'ye dönüştürür.

**Temel özellikler:**

- Frontend ile WebSocket bağlantısı
- Vision, Emotion ve Speech servislerine gRPC bağlantıları
- Veri dönüşümü ve birleştirme
- Hata yönetimi ve loglama

**Nasıl çalışır:**

1. Gateway, WebSocket sunucusu ve gRPC istemcilerini oluşturur
2. Frontend'den gelen görüntü kareleri Vision Service'e iletilir
3. Vision Service'ten gelen yüz verileri Emotion ve Speech Service'lere gönderilir
4. Tüm servislerden gelen yanıtlar birleştirilip WebSocket üzerinden Frontend'e gönderilir

### Vision Service (Python)

Vision Service, görüntü işleme ve yüz tespiti yapar. Tespit ettiği yüzleri diğer servislere iletir.

**Temel özellikler:**

- Yüz tespiti ve takibi
- Yüz landmark noktalarını bulma
- Tespit edilen yüzleri Emotion ve Speech servislerine iletme
- gRPC sunucusu ile hizmet sunma

**Nasıl çalışır:**

1. Gateway'den gelen gRPC isteklerini karşılar
2. Görüntü karesi üzerinde yüz tespiti yapar
3. Tespit edilen her yüzün landmark noktalarını çıkarır ve izleme yapar
4. Tespit edilen yüz verileri Emotion ve Speech servislerine gönderilir

### Emotion Service (Python)

Emotion Service, yüz görüntülerinden duygu analizi yapan servistir.

**Temel özellikler:**

- DeepFace ile gelişmiş duygu analizi
- Bölgesel ve zamansal duygu tespiti
- Kişi bazlı adaptif kalibrasyon
- Duygu geçişlerinde kararlılığı artıran filtreler

**Nasıl çalışır:**

1. Vision Service'ten gelen yüz görüntülerini analiz eder
2. Yüz üzerinde bölgesel analizler yaparak duygu ipuçlarını tespit eder
3. DeepFace destekli derin öğrenme modelleriyle duygu skorları hesaplar
4. Zamansal filtrelerle duygu değişimlerini yumuşatır
5. Tespit edilen duygu ve güven skorunu Gateway'e gönderir

### Speech Detection Service (Python)

Speech Detection Service, yüz landmark noktalarını kullanarak kişinin konuşup konuşmadığını tespit eder.

**Temel özellikler:**

- Landmark noktaları arası oran hesaplama
- Ağız hareketlerinin analizi
- Konuşma durumu ve süre takibi

**Nasıl çalışır:**

1. Vision Service'ten gelen yüz landmark verilerini analiz eder
2. Ağzın açıklığını ve hareketlerini hesaplar
3. Zamansal örüntüleri inceleyerek konuşma durumunu belirler
4. Konuşma durumu ve süresini Gateway'e geri döndürür

## Başlatma Talimatları

### Önkoşullar

- Node.js (v14+)
- Python (v3.8+)
- Temel Python kütüphaneleri: opencv-python, numpy, dlib, deepface, grpcio
- gRPC araçları

### Servisleri Başlatma

1. **Emotion Service'i Başlatma**:

```bash
cd emotion-service
pip install -r requirements.txt
python emotion_server.py
```

2. **Speech Detection Service'i Başlatma**:

```bash
cd speech-service
pip install -r requirements.txt
python speech_server.py
```

3. **Vision Service'i Başlatma**:

```bash
cd vision-service
pip install -r requirements.txt
python vision_server.py
```

4. **Gateway'i Başlatma**:

```bash
cd gateway
npm install
node server.js
```

5. **Frontend'i Başlatma**:

```bash
cd frontend
npm install
npm run dev
```

6. Tarayıcınızda `http://localhost:5173` adresine giderek uygulamaya erişebilirsiniz.

## Ortam Değişkenleri

Her bileşen `.env` dosyaları aracılığıyla yapılandırılabilir:

- **Vision Service**: Yüz tanıma eşik değerleri ve servis bağlantıları
- **Emotion Service**: Duygu analizi parametreleri ve kalibrasyon değerleri
- **Speech Service**: Konuşma tespiti algoritması parametreleri
- **Gateway**: WebSocket ve gRPC bağlantı ayarları
- **Genel**: Servis adresleri ve port numaraları

## Mimari Detaylar

- **Modüler Tasarım**: Her servis, kendi uzmanlık alanında işlem yapacak şekilde ayrılmıştır
- **Duygu Analizi**: DeepFace ile derin öğrenme destekli duygu analizi yapar ve zaman içinde kararlılık sağlamak için gelişmiş filtreler kullanır
- **Yüz Takibi**: Kosinüs benzerliği ile yüz takibi yapar
- **Konuşma Tespiti**: Landmark noktaları arasındaki dinamik ilişkilerin analizi ile konuşma algılaması yapar

## İleriki Gelişmeler

- Çoklu dil desteği
- Gerçek zamanlı analitik grafikler
- Kalabalık ortamlarda performans iyileştirmeleri
- Birden fazla kişinin sosyal etkileşimlerinin analizi
