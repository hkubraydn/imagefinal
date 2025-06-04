"""
Real-time Face Recognition System
Gerçek zamanlı laptop kamerası ile yüz tanıma sistemi
"""

import cv2
import numpy as np
import json
import os
import base64
from utils import get_user_input, save_face_data, load_face_data
import time
import threading
from queue import Queue

class RealTimeFaceRecognition:
    def __init__(self, data_file="face_data.json"):
        """
        Gerçek zamanlı yüz tanıma sistemi başlatıcı.
        """
        self.data_file = data_file
        self.known_faces = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.is_trained = False
        self.last_unknown_face_time = 0
        self.unknown_face_cooldown = 5  # 5 saniye bekleme süresi
        self.last_processed_time = 0
        self.processing_interval = 0.033  # 30 FPS
        self.face_detection_scale = 1.05
        self.face_detection_neighbors = 3
        self.min_face_size = (30, 30)
        self.recognition_threshold = 80
        
        # Yüz takibi için değişkenler
        self.last_face_locations = []
        self.face_tracking_frames = 0
        self.max_tracking_frames = 10
        
        # Yüz stabilizasyonu için değişkenler
        self.face_history = []  # Son tespit edilen yüzlerin geçmişi
        self.history_size = 5  # Kaç frame'lik geçmiş tutulacak
        self.stability_threshold = 0.7  # Yüz stabilizasyonu için eşik değeri
        
        # Asenkron işleme için değişkenler
        self.name_input_queue = Queue()
        self.is_waiting_for_name = False
        self.current_face_roi = None
        
        # Mevcut yüz verilerini yükle
        self.load_known_faces()
        
        # Kamerayı başlat
        self.init_camera()
        
        print(f"Sistem başlatıldı. Kayıtlı yüz sayısı: {len(self.known_faces)}")

    def init_camera(self):
        """Kamerayı başlat ve optimize et."""
        print("Kamera başlatılıyor...")
        self.video_capture = cv2.VideoCapture(0)
        
        if not self.video_capture.isOpened():
            print("Kamera açılamadı. Farklı kamera indeksleri deneniyor...")
            for i in range(1, 4):
                self.video_capture = cv2.VideoCapture(i)
                if self.video_capture.isOpened():
                    print(f"Kamera {i} numaralı indekste bulundu.")
                    break
            else:
                raise Exception("Hiçbir kamera bulunamadı. Lütfen kamera bağlantınızı kontrol edin.")
        
        # Kamera ayarlarını optimize et
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer boyutunu küçült
        
        print("Kamera başarıyla başlatıldı.")

    def load_known_faces(self):
        """Kayıtlı yüz verilerini yükle."""
        try:
            if not os.path.exists(self.data_file):
                print("Yüz verisi dosyası bulunamadı. Yeni dosya oluşturuluyor.")
                return

            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if data and 'faces' in data:
                # Base64 formatındaki görüntüleri numpy dizilerine çevir
                for face_data in data['faces']:
                    face_images = []
                    for img_base64 in face_data['images']:
                        # Base64'ten numpy dizisine çevir
                        img_bytes = base64.b64decode(img_base64)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                        face_images.append(img)
                    
                    self.known_faces.append({
                        'name': face_data['name'],
                        'images': face_images
                    })
                
                if self.known_faces:
                    self.retrain_recognizer()
                print(f"Başarıyla {len(self.known_faces)} yüz yüklendi.")
            else:
                print("Geçerli yüz verisi bulunamadı. Boş veritabanı ile başlanıyor.")
        except Exception as e:
            print(f"Yüz verisi yükleme hatası: {str(e)}")
            print("Boş veritabanı ile başlanıyor.")

    def save_known_faces(self):
        """Yüz verilerini kaydet."""
        try:
            # Yüz görüntülerini base64 formatına çevir
            faces_data = []
            for face_data in self.known_faces:
                face_images = []
                for img in face_data['images']:
                    # Numpy dizisini base64'e çevir
                    _, buffer = cv2.imencode('.png', img)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    face_images.append(img_base64)
                
                faces_data.append({
                    'name': face_data['name'],
                    'images': face_images
                })
            
            data = {
                'faces': faces_data
            }
            
            # JSON formatında kaydet
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"Yüz verileri başarıyla kaydedildi. Toplam yüz: {len(self.known_faces)}")
        except Exception as e:
            print(f"Yüz verisi kaydetme hatası: {str(e)}")

    def retrain_recognizer(self):
        """Yüz tanıyıcıyı yeniden eğit."""
        if not self.known_faces:
            self.is_trained = False
            return
        
        faces = []
        labels = []
        
        for i, face_data in enumerate(self.known_faces):
            for face_img in face_data['images']:
                faces.append(face_img)
                labels.append(i)
        
        if faces:
            self.face_recognizer.train(faces, np.array(labels))
            self.is_trained = True
            print(f"Yüz tanıyıcı {len(faces)} yüz örneği ile eğitildi.")

    def register_new_face(self, face_img, name):
        """Yeni yüz kaydet."""
        # Bu isim zaten var mı kontrol et
        for face_data in self.known_faces:
            if face_data['name'].lower() == name.lower():
                # Mevcut kişi için yeni örnek ekle
                face_data['images'].append(face_img)
                print(f"Mevcut kişi için yeni örnek eklendi: {name}")
                self.retrain_recognizer()
                self.save_known_faces()
                return
        
        # Yeni yüz kaydı oluştur
        new_face = {
            'name': name,
            'images': [face_img]
        }
        self.known_faces.append(new_face)
        
        # Tanıyıcıyı yeniden eğit ve kaydet
        self.retrain_recognizer()
        self.save_known_faces()
        print(f"Yeni yüz kaydedildi: {name}")

    def get_user_input_for_face(self):
        """Yeni yüz için kullanıcıdan isim al (asenkron)."""
        if not self.is_waiting_for_name:
            self.is_waiting_for_name = True
            print("\n" + "="*50)
            print("YENİ YÜZ TESPİT EDİLDİ!")
            print("="*50)
            print("Terminal penceresine geçin ve isim girin.")
            
            # Asenkron isim girişi için thread başlat
            input_thread = threading.Thread(target=self._get_name_input)
            input_thread.daemon = True
            input_thread.start()
        
        # Kuyruktan isim al
        try:
            name = self.name_input_queue.get_nowait()
            self.is_waiting_for_name = False
            return name
        except:
            return None

    def _get_name_input(self):
        """Asenkron isim girişi için yardımcı fonksiyon."""
        name = input("Bu kişinin ismini girin (boş bırakırsanız kayıt yapılmaz): ").strip()
        self.name_input_queue.put(name if name else None)

    def stabilize_face_detection(self, current_faces):
        """
        Yüz tespitini stabilize et ve yanlış pozitifleri filtrele.
        
        Args:
            current_faces: Şu anki frame'de tespit edilen yüzler
            
        Returns:
            list: Stabilize edilmiş yüz konumları
        """
        # NumPy dizisini listeye çevir
        if isinstance(current_faces, np.ndarray):
            current_faces = current_faces.tolist()
        
        if not current_faces:
            # Eğer yüz tespit edilmediyse ve geçmiş varsa, son tespit edilen yüzü kullan
            if self.face_history:
                return self.face_history[-1]
            return []

        # Yeni tespit edilen yüzleri geçmişe ekle
        self.face_history.append(current_faces)
        
        # Geçmiş boyutunu kontrol et
        if len(self.face_history) > self.history_size:
            self.face_history.pop(0)
        
        # Eğer yeterli geçmiş yoksa, mevcut tespitleri kullan
        if len(self.face_history) < 3:
            return current_faces
        
        # Son 3 frame'deki yüzleri karşılaştır
        stable_faces = []
        for face in current_faces:
            x, y, w, h = face[:4]
            face_center = (x + w//2, y + h//2)
            
            # Bu yüzün son frame'lerdeki konumlarını kontrol et
            face_stable = True
            for prev_faces in self.face_history[-3:-1]:
                found_match = False
                for prev_face in prev_faces:
                    px, py, pw, ph = prev_face[:4]
                    prev_center = (px + pw//2, py + ph//2)
                    
                    # İki yüz merkezi arasındaki mesafe
                    distance = np.sqrt((face_center[0] - prev_center[0])**2 + 
                                     (face_center[1] - prev_center[1])**2)
                    
                    # Eğer mesafe belirli bir eşik değerinin altındaysa, aynı yüz
                    if distance < min(w, pw) * 0.5:
                        found_match = True
                        break
                
                if not found_match:
                    face_stable = False
                    break
            
            if face_stable:
                stable_faces.append(face)
        
        return stable_faces

    def process_frame(self, frame):
        """Tek bir frame'i işle."""
        current_time = time.time()
        
        # İşleme sıklığını kontrol et
        if current_time - self.last_processed_time < self.processing_interval:
            # Son tespit edilen yüzleri kullan
            if self.last_face_locations:
                for (x, y, w, h, name, confidence) in self.last_face_locations:
                    self.draw_face_box(frame, x, y, w, h, name, confidence)
            return frame
        
        self.last_processed_time = current_time
        
        # Frame'i küçült (performans için)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Gürültü azaltma
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Yüz tespiti
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.face_detection_scale,
            minNeighbors=self.face_detection_neighbors,
            minSize=self.min_face_size
        )
        
        # Yüz tespitini stabilize et
        stable_faces = self.stabilize_face_detection(faces)
        
        # Yeni yüz konumlarını sakla
        self.last_face_locations = []
        
        # Tespit edilen yüzleri işle
        for face in stable_faces:
            # Koordinatları orijinal boyuta çevir
            x, y, w, h = [int(coord * 2) for coord in face[:4]]
            
            # Yüz bölgesini çıkar
            face_roi = gray[y//2:y//2+h//2, x//2:x//2+w//2]
            face_roi_resized = cv2.resize(face_roi, (100, 100))
            
            name = "Bilinmeyen"
            confidence = 0
            
            if self.is_trained:
                # Yüzü tanımaya çalış
                label, confidence = self.face_recognizer.predict(face_roi_resized)
                
                # Güven eşiği kontrolü
                if confidence < self.recognition_threshold and label < len(self.known_faces):
                    name = self.known_faces[label]['name']
                else:
                    name = "Bilinmeyen"
            
            # Bilinmeyen yüz ise ve cooldown süresi geçtiyse kayıt sor
            if name == "Bilinmeyen" and (current_time - self.last_unknown_face_time) > self.unknown_face_cooldown:
                # Mevcut frame'i göster
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(temp_frame, "YENİ YÜZ - Terminal'e geçin", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow('Yüz Tanıma Sistemi', temp_frame)
                cv2.waitKey(1)
                
                # Yüz ROI'sini sakla
                self.current_face_roi = face_roi_resized
                
                # Asenkron isim girişi
                user_name = self.get_user_input_for_face()
                if user_name:
                    # Aynı isimde kayıt var mı kontrol et
                    is_duplicate = False
                    for face_data in self.known_faces:
                        if face_data['name'].lower() == user_name.lower():
                            is_duplicate = True
                            print(f"Bu isimde bir kayıt zaten var: {user_name}")
                            break
                    
                    if not is_duplicate and self.current_face_roi is not None:
                        self.register_new_face(self.current_face_roi, user_name)
                        name = user_name
                
                self.last_unknown_face_time = current_time
                self.current_face_roi = None
            
            # Yüz konumunu ve bilgilerini sakla
            self.last_face_locations.append((x, y, w, h, name, confidence))
            
            # Yüz çerçevesini çiz
            self.draw_face_box(frame, x, y, w, h, name, confidence)
        
        return frame

    def draw_face_box(self, frame, x, y, w, h, name, confidence):
        """Yüz çevresine çerçeve ve isim çiz."""
        # Renk seçimi
        if name == "Bilinmeyen":
            color = (0, 0, 255)  # Kırmızı
            status = "BILINMEYEN"
        else:
            color = (0, 255, 0)  # Yeşil
            status = "TANINDI"
        
        # Yüz çevresine dikdörtgen çiz (daha kalın çizgi)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # İsim ve durum için arka plan
        label_text = f"{name}"
        if confidence > 0:
            label_text += f" ({confidence:.0f})"
        
        # Metin boyutunu hesapla
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # Daha büyük yazı
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Arka plan dikdörtgeni (daha büyük)
        cv2.rectangle(frame, (x, y-text_height-15), (x+text_width+10, y), color, cv2.FILLED)
        
        # İsim metni
        cv2.putText(frame, label_text, (x+5, y-5), font, font_scale, (255, 255, 255), thickness)
        
        # Durum metni
        cv2.putText(frame, status, (x, y+h+25), font, 0.6, color, 2)

    def add_info_overlay(self, frame):
        """Frame'e bilgi overlay'i ekle."""
        info_lines = [
            f"Kayitli kisi sayisi: {len(self.known_faces)}",
            f"Model durumu: {'Egitildi' if self.is_trained else 'Egitilmedi'}",
            "Cikis icin 'q' basin"
        ]
        
        # Yarı saydam arka plan
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Bilgi metinleri
        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        return frame

    def run_real_time_recognition(self):
        """Ana gerçek zamanlı tanıma döngüsü."""
        print("\n" + "="*60)
        print("GERÇEK ZAMANLI YÜZ TANIMA SİSTEMİ BAŞLATILIYOR")
        print("="*60)
        print("Talimatlar:")
        print("- Kameraya bakın, sistem yüzünüzü otomatik olarak tespit edecek")
        print("- Bilinmeyen yüz tespit edildiğinde terminal'de isim girmeniz istenecek")
        print("- Tanınan yüzler yeşil çerçeve ile gösterilir")
        print("- Bilinmeyen yüzler kırmızı çerçeve ile gösterilir")
        print("- Çıkmak için kamera penceresinde 'q' tuşuna basın")
        print("\nKamera açılıyor...")
        
        try:
            while True:
                # Frame oku
                ret, frame = self.video_capture.read()
                
                if not ret:
                    print("Kameradan frame okunamadı.")
                    break
                
                # Frame'i aynala (selfie etkisi)
                frame = cv2.flip(frame, 1)
                
                # Frame'i işle
                frame = self.process_frame(frame)
                
                # Bilgi overlay'ini ekle
                frame = self.add_info_overlay(frame)
                
                # Frame'i göster
                cv2.imshow('Yüz Tanıma Sistemi', frame)
                
                # 'q' tuşu ile çıkış
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):  # 'r' tuşu ile tanıyıcıyı yeniden eğit
                    if self.known_faces:
                        print("Tanıyıcı yeniden eğitiliyor...")
                        self.retrain_recognizer()
                
        except KeyboardInterrupt:
            print("\nProgram kullanıcı tarafından sonlandırıldı.")
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Kaynakları temizle."""
        print("\nSistem kapatılıyor...")
        if hasattr(self, 'video_capture'):
            self.video_capture.release()
        cv2.destroyAllWindows()
        print("Sistem başarıyla kapatıldı.")

    def __del__(self):
        """Destructor."""
        self.cleanup()

def main():
    """Ana fonksiyon."""
    print("OpenCV Gerçek Zamanlı Yüz Tanıma Sistemi")
    print("==========================================")
    
    try:
        # Sistemi başlat
        face_system = RealTimeFaceRecognition()
        
        # Gerçek zamanlı tanımayı başlat
        face_system.run_real_time_recognition()
        
    except Exception as e:
        print(f"Sistem başlatılamadı: {str(e)}")
        print("\nMuhtemel çözümler:")
        print("1. Kamera bağlantısını kontrol edin")
        print("2. Başka uygulamaların kamerayı kullanmadığından emin olun")
        print("3. Kamera sürücülerini kontrol edin")

if __name__ == "__main__":
    main()