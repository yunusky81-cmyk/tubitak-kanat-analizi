"""
Deneme videosu üretir: gerçek kuş/robot videon olmadan kodu test etmek için.
  robot.mp4 : sabit 8 Hz kanat çırpma, düz rota
  kus.mp4   : frekansı sürekli değişen çırpma + süzülme anları, dalgalı rota

Kullanım: python test_videosu_uret.py
"""
import cv2
import numpy as np

FPS, SURE, G, Y = 60, 8, 640, 360


def ciz(kare, x, y, aciklik):
    cv2.ellipse(kare, (int(x), int(y)), (14, 8), 0, 0, 360, (40, 40, 40), -1)   # gövde
    kanat = int(6 + 34 * aciklik)                                                # kanat açıklığı
    cv2.ellipse(kare, (int(x), int(y)), (kanat, 4), 0, 0, 360, (40, 40, 40), -1)


def uret(dosya, robot):
    yazici = cv2.VideoWriter(dosya, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (G, Y))
    rng = np.random.default_rng(1)
    faz, frek = 0.0, 8.0
    for i in range(FPS * SURE):
        t = i / FPS
        kare = np.full((Y, G, 3), 225, np.uint8)
        if robot:
            frek = 8.0
            x, y = 40 + t * 70, 180
        else:
            frek = float(np.clip(frek + rng.normal(0, 0.35), 3, 14))   # frekans rastgele kayar
            x, y = 40 + t * 70, 180 + 40 * np.sin(t * 1.3)
        suzulme = (not robot) and (int(t * 2) % 5 == 4)                  # ara ara süzülme
        if not suzulme:
            faz += 2 * np.pi * frek / FPS
        ciz(kare, x, y, 0.5 + 0.5 * np.sin(faz))
        yazici.write(kare)
    yazici.release()
    print("Üretildi:", dosya)


if __name__ == "__main__":
    uret("robot.mp4", robot=True)
    uret("kus.mp4", robot=False)
