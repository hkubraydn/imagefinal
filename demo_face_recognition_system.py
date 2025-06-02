"""
Demo Face Recognition System
OpenCV kullanarak yüz tanıma sistemi demonstrasyonu
"""

import cv2
import numpy as np
import pickle
import os
from utils import get_user_input, save_face_data, load_face_data

class DemoFaceRecognitionSystem:
    def __init__(self, data_file="face_data.pkl"):
        """
        Demo Face Recognition System başlatıcı.
        """
        self.data_file = data_file
        self.known_faces = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.is_trained = False
        
        # Mevcut yüz verilerini yükle
        self.load_known_faces()
        
        print(f"Demo sistem başlatıldı. Kayıtlı yüz sayısı: {len(self.known_faces)}")

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

    def create_sample_images(self):
        """Demo için örnek yüz görüntüleri oluştur."""
        print("\nDemo için örnek yüzler oluşturuluyor...")
        
        # Basit yüz benzeri görüntüler oluştur
        sample_faces = []
        
        # Örnek yüz 1
        face1 = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        # Yüz benzeri desen ekle
        cv2.circle(face1, (30, 30), 5, 255, -1)  # Sol göz
        cv2.circle(face1, (70, 30), 5, 255, -1)  # Sağ göz
        cv2.ellipse(face1, (50, 70), (15, 5), 0, 0, 180, 255, 2)  # Ağız
        sample_faces.append(("Ahmet", face1))
        
        # Örnek yüz 2
        face2 = np.random.randint(40, 180, (100, 100), dtype=np.uint8)
        cv2.circle(face2, (25, 35), 6, 255, -1)  # Sol göz
        cv2.circle(face2, (75, 35), 6, 255, -1)  # Sağ göz
        cv2.ellipse(face2, (50, 75), (20, 8), 0, 0, 180, 255, 2)  # Ağız
        sample_faces.append(("Ayşe", face2))
        
        # Örnek yüz 3
        face3 = np.random.randint(60, 220, (100, 100), dtype=np.uint8)
        cv2.circle(face3, (35, 25), 4, 255, -1)  # Sol göz
        cv2.circle(face3, (65, 25), 4, 255, -1)  # Sağ göz
        cv2.ellipse(face3, (50, 65), (12, 4), 0, 0, 180, 255, 2)  # Ağız
        sample_faces.append(("Mehmet", face3))
        
        return sample_faces

    def demo_face_registration(self):
        """Demo yüz kayıt sistemi."""
        print("\n" + "="*60)
        print("YÜZ KAYIT SİSTEMİ DEMO")
        print("="*60)
        
        sample_faces = self.create_sample_images()
        
        for name, face_img in sample_faces:
            print(f"\n{name} için yüz kaydediliyor...")
            self.register_new_face(face_img, name)
            
            # Görüntüyü göster
            display_img = cv2.resize(face_img, (200, 200))
            cv2.imshow(f'Kayıt: {name}', display_img)
            cv2.waitKey(1000)  # 1 saniye göster
            cv2.destroyAllWindows()

    def demo_face_recognition(self):
        """Demo yüz tanıma sistemi."""
        print("\n" + "="*60)
        print("YÜZ TANIMA SİSTEMİ DEMO")
        print("="*60)
        
        if not self.is_trained:
            print("Henüz eğitilmiş model yok. Önce yüz kayıt demosu çalıştırın.")
            return
        
        # Test görüntüleri oluştur (kayıtlı yüzlere benzer)
        test_faces = []
        
        # Ahmet'e benzer yüz
        test_face1 = np.random.randint(45, 195, (100, 100), dtype=np.uint8)
        cv2.circle(test_face1, (32, 32), 5, 255, -1)
        cv2.circle(test_face1, (68, 32), 5, 255, -1)
        cv2.ellipse(test_face1, (50, 72), (16, 6), 0, 0, 180, 255, 2)
        test_faces.append(("Test 1 (Ahmet benzeri)", test_face1))
        
        # Ayşe'ye benzer yüz
        test_face2 = np.random.randint(35, 175, (100, 100), dtype=np.uint8)
        cv2.circle(test_face2, (27, 37), 6, 255, -1)
        cv2.circle(test_face2, (73, 37), 6, 255, -1)
        cv2.ellipse(test_face2, (50, 77), (18, 7), 0, 0, 180, 255, 2)
        test_faces.append(("Test 2 (Ayşe benzeri)", test_face2))
        
        # Bilinmeyen yüz
        unknown_face = np.random.randint(80, 240, (100, 100), dtype=np.uint8)
        cv2.circle(unknown_face, (20, 20), 3, 255, -1)
        cv2.circle(unknown_face, (80, 20), 3, 255, -1)
        cv2.line(unknown_face, (30, 80), (70, 80), 255, 2)
        test_faces.append(("Test 3 (Bilinmeyen)", unknown_face))
        
        # Her test yüzünü tanı
        for test_name, test_face in test_faces:
            print(f"\n{test_name} tanınıyor...")
            
            if self.is_trained:
                label, confidence = self.face_recognizer.predict(test_face)
                
                if confidence < 100 and label < len(self.known_faces):
                    recognized_name = self.known_faces[label]['name']
                    print(f"✓ Tanınan: {recognized_name} (Güven: {confidence:.1f})")
                    result_text = f"Tanınan: {recognized_name}"
                    color = (0, 255, 0)  # Yeşil
                else:
                    print(f"✗ Bilinmeyen yüz (Güven: {confidence:.1f})")
                    result_text = "Bilinmeyen"
                    color = (0, 0, 255)  # Kırmızı
            else:
                result_text = "Model eğitilmedi"
                color = (0, 0, 255)
            
            # Sonucu görüntüle
            display_img = cv2.resize(test_face, (300, 300))
            display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            
            # Sonuç metnini ekle
            cv2.putText(display_img_bgr, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_img_bgr, test_name, (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Yuz Tanima Sonucu', display_img_bgr)
            cv2.waitKey(2000)  # 2 saniye göster
            cv2.destroyAllWindows()

    def interactive_demo(self):
        """İnteraktif demo menüsü."""
        while True:
            print("\n" + "="*60)
            print("YÜZ TANIMA SİSTEMİ - İNTERAKTİF DEMO")
            print("="*60)
            print(f"Kayıtlı yüz sayısı: {len(self.known_faces)}")
            print(f"Model durumu: {'Eğitildi' if self.is_trained else 'Eğitilmedi'}")
            print("\nSeçenekler:")
            print("1. Yüz kayıt demosu")
            print("2. Yüz tanıma demosu")
            print("3. Kayıtlı yüzleri listele")
            print("4. Verileri temizle")
            print("5. Çıkış")
            
            choice = input("\nSeçiminizi yapın (1-5): ").strip()
            
            if choice == "1":
                self.demo_face_registration()
            elif choice == "2":
                self.demo_face_recognition()
            elif choice == "3":
                self.list_known_faces()
            elif choice == "4":
                self.clear_data()
            elif choice == "5":
                print("Demo sonlandırılıyor...")
                break
            else:
                print("Geçersiz seçim. Lütfen 1-5 arası bir sayı girin.")

    def list_known_faces(self):
        """Kayıtlı yüzleri listele."""
        print("\n" + "="*40)
        print("KAYITLI YÜZLER")
        print("="*40)
        
        if not self.known_faces:
            print("Henüz kayıtlı yüz yok.")
        else:
            for i, face_data in enumerate(self.known_faces, 1):
                print(f"{i}. {face_data['name']} - {len(face_data['images'])} örnek")

    def clear_data(self):
        """Tüm verileri temizle."""
        confirmation = input("Tüm kayıtlı yüzleri silmek istediğinizden emin misiniz? (evet/hayır): ").strip().lower()
        
        if confirmation in ['evet', 'e', 'yes', 'y']:
            self.known_faces = []
            self.is_trained = False
            
            # Dosyayı da sil
            if os.path.exists(self.data_file):
                os.remove(self.data_file)
            
            print("Tüm veriler temizlendi.")
        else:
            print("İşlem iptal edildi.")

def main():
    """Ana fonksiyon."""
    print("OpenCV Yüz Tanıma Sistemi Demo")
    print("Bu demo, gerçek kamera olmadan yüz tanıma sisteminin nasıl çalıştığını gösterir.")
    
    demo_system = DemoFaceRecognitionSystem()
    demo_system.interactive_demo()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()