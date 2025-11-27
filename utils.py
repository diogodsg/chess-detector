import cv2
import numpy as np

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
