# Kanat Çırpma Ritmi Analizi — Kuş mu, Robot mu?

TÜBİTAK 2204 projesi için prototip. Kamerada uçan nesnenin kanat çırpma ritmini (FFT) analiz ederek
gerçek kuşu kuş şeklindeki robottan (ornitopter) ayırt etmeye çalışır.

- Robot: sabit frekans, yüksek düzenlilik
- Kuş: değişken frekans, süzülme, düzensiz ritim

## Kurulum
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Kullanım
```
.venv/bin/python test_videosu_uret.py              # deneme videoları (robot.mp4, kus.mp4)
.venv/bin/python kanat_analizi.py --kaynak robot.mp4
.venv/bin/python kanat_analizi.py --kaynak 0       # bilgisayar kamerası, q ile çıkış
```
Ölçümler `olcumler.csv` dosyasına kaydedilir. En iyi sonuç için telefonda ağır çekim (120/240 fps) video kullan.
