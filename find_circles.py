import cv2
import numpy as np

CAMINHO_IMAGEM = r"C:\Users\User\Desktop\PDI_Current_FP\chess-detector\resultados\6_homografia.jpg"

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




def main():
   


if __name__ == "__main__":
    main()
