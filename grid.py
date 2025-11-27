import cv2
import numpy as np
from utils import ordenar_cantos

def validar_quadrilatero_tabuleiro(cantos):
    """
    Verifica se 4 pontos formam um quadrilátero válido para um tabuleiro:
    - Área mínima
    - Lados opostos aproximadamente paralelos
    - Proporção aproximadamente quadrada (aspect ratio)
    - Ângulos internos razoáveis
    """
    p1, p2, p3, p4 = cantos  # tl, tr, bl, br
    
    # Vetores dos lados
    top = np.array([p2[0] - p1[0], p2[1] - p1[1]])      # top-left -> top-right
    bottom = np.array([p4[0] - p3[0], p4[1] - p3[1]])   # bottom-left -> bottom-right
    left = np.array([p3[0] - p1[0], p3[1] - p1[1]])     # top-left -> bottom-left
    right = np.array([p4[0] - p2[0], p4[1] - p2[1]])    # top-right -> bottom-right
    
    # Comprimentos dos lados
    len_top = np.linalg.norm(top)
    len_bottom = np.linalg.norm(bottom)
    len_left = np.linalg.norm(left)
    len_right = np.linalg.norm(right)
    
    if min(len_top, len_bottom, len_left, len_right) < 50:
        return False
    
    # Verificar proporção dos lados (aspect ratio próximo de 1 para tabuleiro quadrado)
    ratio_horizontal = max(len_top, len_bottom) / max(min(len_top, len_bottom), 1)
    ratio_vertical = max(len_left, len_right) / max(min(len_left, len_right), 1)
    
    if ratio_horizontal > 2.0 or ratio_vertical > 2.0:
        return False
    
    # Verificar paralelismo: ângulo entre lados opostos
    def angulo_entre_vetores(v1, v2):
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_ang = np.clip(cos_ang, -1, 1)
        return np.abs(np.arccos(cos_ang)) * 180 / np.pi
    
    ang_top_bottom = angulo_entre_vetores(top, bottom)
    ang_left_right = angulo_entre_vetores(left, right)
    
    # Lados opostos devem ser aproximadamente paralelos (< 30°)
    if ang_top_bottom > 30 or ang_left_right > 30:
        return False
    
    # Verificar que não é muito distorcido (ângulo entre lados adjacentes)
    ang_corner = angulo_entre_vetores(top, left)
    if ang_corner < 30 or ang_corner > 150:  # Deve ser próximo de 90°
        return False
    
    # Área mínima
    area = 0.5 * abs(np.cross(top, left)) + 0.5 * abs(np.cross(bottom, right))
    if area < 10000:
        return False
    
    return True

def calcular_erro_grid(cantos, intersecoes, n_casas=8):
    """
    Calcula o erro de um modelo de tabuleiro.
    Os 4 cantos devem corresponder às posições (0,0), (8,0), (0,8), (8,8) do grid.
    Retorna: (erro_total, encontrados, nao_encontrados, inliers, grid_pontos)
    - erro_total: soma das distâncias euclidianas de todas as interseções
    - encontrados: quantidade de interseções próximas de onde deveriam estar
    - nao_encontrados: quantidade de posições do grid sem interseção
    """
    cantos = ordenar_cantos(cantos)  # [tl, tr, bl, br]
    tam = 800
    casa = tam / n_casas
    n_pontos_grid = n_casas + 1  # 9x9 = 81 pontos
    
    src = np.float32(cantos)
    dst = np.float32([[0, 0], [tam, 0], [0, tam], [tam, tam]])
    
    try:
        matriz = cv2.getPerspectiveTransform(src, dst)
    except:
        return float('inf'), 0, 81, [], None
    
    # Transformar todas as interseções
    pts = np.float32(intersecoes).reshape(-1, 1, 2)
    pts_transformados = cv2.perspectiveTransform(pts, matriz).reshape(-1, 2)
    
    grid_pontos = {}
    threshold = casa * 0.3
    soma_distancias = 0.0
    
    # Adicionar os 4 cantos como posições fixas (distância = 0)
    grid_pontos[(0, 0)] = (cantos[0], 0)
    grid_pontos[(n_casas, 0)] = (cantos[1], 0)
    grid_pontos[(0, n_casas)] = (cantos[2], 0)
    grid_pontos[(n_casas, n_casas)] = (cantos[3], 0)
    
    for gx in range(n_pontos_grid):
        for gy in range(n_pontos_grid):
            # Pular os cantos (já foram adicionados)
            if (gx, gy) in [(0, 0), (n_casas, 0), (0, n_casas), (n_casas, n_casas)]:
                continue
                
            esperado_x = gx * casa
            esperado_y = gy * casa
            
            menor_dist = float('inf')
            melhor_idx = -1
            
            for i, (x, y) in enumerate(pts_transformados):
                dist = np.sqrt((x - esperado_x)**2 + (y - esperado_y)**2)
                if dist < menor_dist:
                    menor_dist = dist
                    melhor_idx = i
            
            if menor_dist < threshold:
                grid_pontos[(gx, gy)] = (intersecoes[melhor_idx], menor_dist)
                soma_distancias += menor_dist
    
    # Contar encontrados e não encontrados
    total_posicoes = n_pontos_grid * n_pontos_grid  # 81
    encontrados = len(grid_pontos)
    nao_encontrados = total_posicoes - encontrados
    
    # Penalidade para posições não encontradas (adiciona threshold para cada vazia)
    penalidade = nao_encontrados * threshold
    erro_total = soma_distancias + penalidade
    
    # Extrair inliers
    inliers = [ponto for ponto, dist in grid_pontos.values()]
    
    # Verificar distribuição mínima
    colunas_ocupadas = set(k[0] for k in grid_pontos.keys())
    linhas_ocupadas = set(k[1] for k in grid_pontos.keys())
    
    if len(colunas_ocupadas) < 5 or len(linhas_ocupadas) < 5:
        return float('inf'), 0, 81, [], None
    
    if encontrados < 25:
        return float('inf'), 0, 81, [], None
    
    return erro_total, encontrados, nao_encontrados, inliers, grid_pontos
