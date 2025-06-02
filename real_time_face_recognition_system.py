"""
Real-time Face Recognition System
Gerçek zamanlı laptop kamerası ile yüz tanıma sistemi
"""

import cv2
import numpy as np
import pickle
import os
from utils import get_user_input, save_face_data, load_face_data

class RealTimeFaceRecognition:
    def __init__(self, data_file="face_data.pkl"):
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
        
        # Mevcut yüz verilerini yükle
        self.load_known_faces()
        
        # Kamerayı başlat
        self.init_camera()
        
        print(f"Sistem başlatıldı. Kayıtlı yüz sayısı: {len(self.known_faces)}")

    def init_camera(self):
        """Kamerayı başlat."""
        print("Kamera başlatılıyor...")
        self.video_capture = cv2.VideoCapture(0)
        
        if not self.video_capture.isOpened():
            print("Kamera açılamadı. Farklı kamera indeksleri deneniyor...")
            # Farklı kamera indekslerini dene
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
        
        print("Kamera başarıyla başlatıldı.")

    def load_known_faces(self):
        """Kayıtlı yüz verilerini yükle."""
        try:
            data = load_face_data(self.data_file)
            if data:
                self.known_faces = data.get('faces', [])
                if self.known_faces:
                    self.retrain_recognizer()
                print(f"Başarıyla {len(self.known_faces)} yüz yüklendi.")
            else:
                print("Mevcut yüz verisi bulunamadı. Boş veritabanı ile başlanıyor.")
        except Exception as e:
            print(f"Yüz verisi yükleme hatası: {str(e)}")
            print("Boş veritabanı ile başlanıyor.")

    def save_known_faces(self):
        """Yüz verilerini kaydet."""
        try:
            data = {
                'faces': self.known_faces
            }
            save_face_data(data, self.data_file)
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
        """Yeni yüz için kullanıcıdan isim al."""
        print("\n" + "="*50)
        print("YENİ YÜZ TESPİT EDİLDİ!")
        print("="*50)
        print("Terminal penceresine geçin ve isim girin.")
        
        # Kullanıcıdan isim al
        name = input("Bu kişinin ismini girin (boş bırakırsanız kayıt yapılmaz): ").strip()
        
        if name:
            return name
        else:
            print("İsim girilmedi. Yüz kaydedilmedi.")
            return None

    def process_frame(self, frame):
        """Tek bir frame'i işle."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        for (x, y, w, h) in faces:
            # Yüz bölgesini çıkar
            face_roi = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (100, 100))
            
            name = "Bilinmeyen"
            confidence = 0
            
            if self.is_trained:
                # Yüzü tanımaya çalış
                label, confidence = self.face_recognizer.predict(face_roi_resized)
                
                # Güven eşiği (düşük değer daha iyi)
                if confidence < 80 and label < len(self.known_faces):
                    name = self.known_faces[label]['name']
                else:
                    name = "Bilinmeyen"
            
            # Bilinmeyen yüz ise ve cooldown süresi geçtiyse kayıt sor
            if name == "Bilinmeyen" and (current_time - self.last_unknown_face_time) > self.unknown_face_cooldown:
                # Mevcut frame'i göster
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(temp_frame, "YENi YUZ - Terminal'e gecin", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow('Yuz Tanima Sistemi', temp_frame)
                cv2.waitKey(1)
                
                user_name = self.get_user_input_for_face()
                if user_name:
                    self.register_new_face(face_roi_resized, user_name)
                    name = user_name
                
                self.last_unknown_face_time = current_time
            
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
        
        # Yüz çevresine dikdörtgen çiz
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # İsim ve durum için arka plan
        label_text = f"{name}"
        if confidence > 0:
            label_text += f" ({confidence:.0f})"
        
        # Metin boyutunu hesapla
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Arka plan dikdörtgeni
        cv2.rectangle(frame, (x, y-text_height-10), (x+text_width+10, y), color, cv2.FILLED)
        
        # İsim metni
        cv2.putText(frame, label_text, (x+5, y-5), font, font_scale, (255, 255, 255), thickness)
        
        # Durum metni
        cv2.putText(frame, status, (x, y+h+20), font, 0.5, color, 1)

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
        
        frame_count = 0
        process_every_n_frames = 3  # Her 3 frame'de bir işle (performans için)
        
        try:
            while True:
                # Frame oku
                ret, frame = self.video_capture.read()
                
                if not ret:
                    print("Kameradan frame okunamadı.")
                    break
                
                # Frame'i aynala (selfie etkisi)
                frame = cv2.flip(frame, 1)
                
                # Yüz tanıma işlemini her 3 frame'de bir yap
                if frame_count % process_every_n_frames == 0:
                    frame = self.process_frame(frame)
                
                # Bilgi overlay'ini ekle
                frame = self.add_info_overlay(frame)
                
                # Frame'i göster
                cv2.imshow('Yuz Tanima Sistemi', frame)
                
                # 'q' tuşu ile çıkış
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):  # 'r' tuşu ile tanıyıcıyı yeniden eğit
                    if self.known_faces:
                        print("Tanıyıcı yeniden eğitiliyor...")
                        self.retrain_recognizer()
                
                frame_count += 1
                
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