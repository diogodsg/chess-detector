import cv2
import numpy as np


# blur forte para estimar fundo (tabuleiro / iluminação)
BACKGROUND_SIGMA = 35

# blur leve antes do Hough/Canny
BLUR_HOUGH_KERNEL = 7
BLUR_HOUGH_SIGMA = 1.5

# HoughCircles (tamanho das peças ~ raio 25–45 px)
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 80     # ~1 casa de distância entre centros
HOUGH_PARAM2 = 20       # menor -> mais círculos (e mais falsos)
MIN_RADIUS = 30
MAX_RADIUS = 45


def desenhar_reta(img, rho, theta, cor=(0, 255, 0), thickness=4):
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    pts = [(int(x0 + 10000 * (-b)), int(y0 + 10000 * a)),
           (int(x0 - 10000 * (-b)), int(y0 - 10000 * a))]
    cv2.line(img, pts[0], pts[1], cor, thickness)

def calcular_intersecao(rho1, theta1, rho2, theta2):
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    det = a1 * b2 - a2 * b1
    
    if abs(det) < 1e-10:
        return None
    
    x = (b2 * rho1 - b1 * rho2) / det
    y = (a1 * rho2 - a2 * rho1) / det
    return (int(x), int(y))

def ordenar_cantos(cantos):
    """Ordena 4 pontos em: top-left, top-right, bottom-left, bottom-right"""
    cantos = sorted(cantos, key=lambda p: p[1])  # Ordenar por Y
    top = sorted(cantos[:2], key=lambda p: p[0])  # Top: ordenar por X
    bottom = sorted(cantos[2:], key=lambda p: p[0])  # Bottom: ordenar por X
    return [top[0], top[1], bottom[0], bottom[1]]



def auto_canny(img_gray,
               proporcao_fortes=0.05,   # ~5% dos pixels como bordas fortes
               low_high_ratio=0.4,      # low = 40% de high
               sobel_ksize=3):
    """
    Canny com limiares automáticos, na linha do slide:
    - calcula o gradiente com Sobel
    - escolhe o limiar alto por percentil da magnitude
    - limiar baixo = fração do alto
    """
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    mag = cv2.magnitude(gx, gy)

    # normaliza para 0..255 e pega só magnitudes > 0
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_flat = mag_norm.ravel()
    mag_flat = mag_flat[mag_flat > 0]

    if len(mag_flat) == 0:
        # fallback se der algo muito estranho
        low, high = 50, 100
        edges = cv2.Canny(img_gray, low, high)
        return edges, low, high

    # percentil para o limiar alto
    perc = 100 * (1.0 - proporcao_fortes)
    high = np.percentile(mag_flat, perc)
    low = low_high_ratio * high

    edges = cv2.Canny(img_gray, int(low), int(high))
    return edges, int(low), int(high)


def detecta_circulos(img_base, canny_high):
    """Roda HoughCircles e devolve uma lista [(x, y, r), ...] em int."""
    circles = cv2.HoughCircles(
        img_base,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=canny_high,      # usa o limiar alto do Canny
        param2=HOUGH_PARAM2,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )

    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype(int)
    return [(x, y, r) for (x, y, r) in circles]


def tira_circulos_fantasma(lista1, dist2_max=100**2):
    """Remove duplicados (centros muito próximos)."""

    if len(lista1) < 20:
        return []

    finais = []

    for (x, y, r) in lista1:
        fantasma = False
        for (x2, y2, r2) in finais:
            if (x - x2) ** 2 + (y - y2) ** 2 < dist2_max:
                fantasma = True
                break
        if not fantasma:
            finais.append((x, y, r))

    return finais