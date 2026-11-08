"""
Kanat Çırpma Ritmi Analizi — Kuş mu, Robot mu?
TÜBİTAK 2204 projesi için prototip.

Fikir:
  - Gerçek kuşun kanat çırpışı düzensizdir (frekans sürekli değişir, süzülür).
  - Ornitopter (kuş robot) motorla çırptığı için frekansı sabit ve düzenlidir.
  Kamerada hareket eden nesnenin alanı kanat açılıp kapandıkça büyüyüp küçülür.
  Bu "alan sinyalinin" frekans analizini (FFT) yapıp düzenliliğini ölçüyoruz.

Kullanım:
  python kanat_analizi.py --kaynak video.mp4
  python kanat_analizi.py --kaynak 0          (bilgisayar kamerası)
  q tuşu: çıkış
"""

import argparse
import collections
import csv
import time

import cv2
import numpy as np


def argumanlari_oku():
    p = argparse.ArgumentParser(description="Kanat çırpma ritmi ile kuş/robot ayırt etme")
    p.add_argument("--kaynak", default="0", help="Video dosyası yolu ya da kamera numarası (0)")
    p.add_argument("--pencere", type=float, default=2.0, help="Analiz penceresi (saniye)")
    p.add_argument("--min-alan", type=int, default=150, help="Bundan küçük hareketleri yok say (piksel)")
    p.add_argument("--fps", type=float, default=0, help="Kaynağın FPS'i okunamazsa elle gir")
    p.add_argument("--duzenlilik-esik", type=float, default=0.45,
                   help="Spektrum tepe oranı bu değerin üstündeyse ritim 'düzenli' sayılır")
    p.add_argument("--sapma-esik", type=float, default=0.6,
                   help="Frekans sapması (Hz) bunun altındaysa ritim 'sabit' sayılır")
    p.add_argument("--csv", default="olcumler.csv", help="Ölçümlerin kaydedileceği dosya")
    p.add_argument("--goster", action="store_true", default=True)
    p.add_argument("--gizle", dest="goster", action="store_false", help="Pencere açmadan çalış")
    return p.parse_args()


def baskin_frekans(sinyal, fps, fmin=2.0):
    """Sinyaldeki en güçlü titreşim frekansını ve ne kadar 'tek tonlu' olduğunu döndürür.

    duzenlilik: 0..1 arası. 1'e yakınsa enerji tek bir frekansta toplanmış (makine gibi),
    0'a yakınsa enerji birçok frekansa dağılmış (doğal, düzensiz hareket).
    """
    x = np.asarray(sinyal, dtype=float)
    x = x - x.mean()
    if np.allclose(x, 0):
        return None, 0.0
    x = x * np.hanning(len(x))                     # pencere kenarlarındaki sıçramaları yumuşatır
    guc = np.abs(np.fft.rfft(x)) ** 2              # güç spektrumu
    frek = np.fft.rfftfreq(len(x), d=1.0 / fps)
    bant = (frek >= fmin) & (frek <= fps / 2 * 0.9)  # Nyquist sınırının biraz altı
    if not bant.any():
        return None, 0.0
    g, f = guc[bant], frek[bant]
    i = int(g.argmax())
    tepe = g[max(i - 1, 0): i + 2].sum()           # tepe ve iki komşusu
    return float(f[i]), float(tepe / g.sum())


def rota_duzlugu(noktalar):
    """1'e yakın: dümdüz rota (planlı). 0'a yakın: zikzak/rastgele rota."""
    if len(noktalar) < 3:
        return None
    p = np.asarray(noktalar, dtype=float)
    p = p[:: max(1, len(p) // 10)]                 # seyrelt: kanat titremesi rotayı bozmasın
    yol = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    if yol < 1e-6:
        return None
    return float(np.linalg.norm(p[-1] - p[0]) / yol)


def karar_ver(duzenlilik, sapma, frek_sayisi, a):
    if frek_sayisi < 5:
        return "ANALIZ EDILIYOR", (200, 200, 200)
    if duzenlilik >= a.duzenlilik_esik and sapma <= a.sapma_esik:
        return "ROBOT OLABILIR", (0, 0, 255)
    return "KUS OLABILIR", (0, 200, 0)


def main():
    a = argumanlari_oku()
    kaynak = int(a.kaynak) if a.kaynak.isdigit() else a.kaynak
    cap = cv2.VideoCapture(kaynak)
    if not cap.isOpened():
        raise SystemExit(f"Kaynak açılamadı: {a.kaynak}")

    fps = a.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    pencere_kare = max(16, int(fps * a.pencere))
    print(f"FPS: {fps:.1f} | Analiz penceresi: {pencere_kare} kare | "
          f"Ölçülebilen en yüksek frekans: {fps / 2:.1f} Hz")

    arka_plan = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)
    cekirdek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    alanlar = collections.deque(maxlen=pencere_kare)
    merkezler = collections.deque(maxlen=pencere_kare)
    frek_gecmisi = collections.deque(maxlen=10)
    kayip_kare, kare_no = 0, 0
    frek = duzenlilik = sapma = duzluk = None
    etiket, renk = "HEDEF YOK", (200, 200, 200)

    dosya = open(a.csv, "w", newline="", encoding="utf-8")
    yazici = csv.writer(dosya)
    yazici.writerow(["kare", "zaman_s", "frekans_hz", "duzenlilik", "frekans_sapmasi_hz", "rota_duzlugu", "karar"])

    while True:
        ok, kare = cap.read()
        if not ok:
            break
        kare_no += 1

        # 1) Hareket eden pikselleri bul
        maske = arka_plan.apply(kare)
        maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, cekirdek)
        maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, cekirdek, iterations=2)
        konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        konturlar = [k for k in konturlar if cv2.contourArea(k) >= a.min_alan]

        # 2) En büyük hareketli nesneyi hedef kabul et
        if konturlar:
            hedef = max(konturlar, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(hedef)
            alanlar.append(cv2.contourArea(hedef))
            merkezler.append((x + w / 2, y + h / 2))
            kayip_kare = 0
            if a.goster:
                cv2.rectangle(kare, (x, y), (x + w, y + h), renk, 2)
        else:
            kayip_kare += 1
            if alanlar and kayip_kare < fps * 0.3:
                alanlar.append(alanlar[-1])        # kısa kayıplarda son değeri tekrarla
            elif kayip_kare >= fps * 0.3:           # hedef gitti, sıfırla
                alanlar.clear(); merkezler.clear(); frek_gecmisi.clear()
                etiket, renk = "HEDEF YOK", (200, 200, 200)

        # 3) Pencere dolunca, her 1/4 pencerede bir analiz yap
        if len(alanlar) == pencere_kare and kare_no % max(1, pencere_kare // 4) == 0:
            frek, duzenlilik = baskin_frekans(alanlar, fps)
            if frek is not None:
                frek_gecmisi.append(frek)
                sapma = float(np.std(frek_gecmisi))
                duzluk = rota_duzlugu(merkezler)
                etiket, renk = karar_ver(duzenlilik, sapma, len(frek_gecmisi), a)
                yazici.writerow([kare_no, round(kare_no / fps, 2), round(frek, 2), round(duzenlilik, 3),
                                 round(sapma, 3), None if duzluk is None else round(duzluk, 3), etiket])

        # 4) Ekrana yaz
        if a.goster:
            satirlar = [etiket]
            if frek is not None and alanlar:
                satirlar += [f"Kanat frekansi: {frek:.1f} Hz",
                             f"Duzenlilik: {duzenlilik:.2f}",
                             f"Frekans sapmasi: {sapma:.2f} Hz"]
                if duzluk is not None:
                    satirlar.append(f"Rota duzlugu: {duzluk:.2f}")
            for i, s in enumerate(satirlar):
                cv2.putText(kare, s, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            renk if i == 0 else (255, 255, 255), 2)
            cv2.imshow("Kanat Analizi", kare)
            bekle = 1 if isinstance(kaynak, int) else max(1, int(1000 / fps))  # video gerçek hızda oynasın
            if cv2.waitKey(bekle) & 0xFF == ord("q"):
                break

    dosya.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Bitti. Ölçümler '{a.csv}' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
