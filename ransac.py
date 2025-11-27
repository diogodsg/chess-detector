import cv2
import numpy as np
from utils import calcular_intersecao, ordenar_cantos

def ransac_encontrar_tabuleiro(linhas_h, linhas_v, intersecoes_reais, n_iter=2000):
    """
    Implementação do RANSAC baseado em Grid do artigo "Determining Chess Game State from an Image".
    """
    if len(linhas_h) < 2 or len(linhas_v) < 2:
        return None, [], None

    melhor_H = None
    melhor_inliers_count = -1
    
    pts_reais = np.array(intersecoes_reais, dtype=np.float32).reshape(-1, 1, 2)
    
    print(f"RANSAC Grid: Iniciando com {len(linhas_h)} linhas H e {len(linhas_v)} linhas V...")
    
    for i in range(n_iter):
        # 1. Amostrar 4 linhas (2 H, 2 V) para formar um retângulo
        idx_h = np.random.choice(len(linhas_h), 2, replace=False)
        idx_v = np.random.choice(len(linhas_v), 2, replace=False)
        
        h1, h2 = linhas_h[idx_h[0]], linhas_h[idx_h[1]]
        v1, v2 = linhas_v[idx_v[0]], linhas_v[idx_v[1]]
        
        # Calcular os 4 cantos desse retângulo
        p1 = calcular_intersecao(*h1, *v1)
        p2 = calcular_intersecao(*h1, *v2)
        p3 = calcular_intersecao(*h2, *v1)
        p4 = calcular_intersecao(*h2, *v2)
        
        if not all([p1, p2, p3, p4]): continue
        
        cantos_amostra = ordenar_cantos([p1, p2, p3, p4])
        src_pts = np.array(cantos_amostra, dtype=np.float32)
        
        # 2. Iterar sobre possíveis tamanhos de grid (sx, sy) de 1 a 8
        # Otimização: testar apenas alguns tamanhos mais prováveis ou todos se rápido
        for sx in range(1, 6): # Testar até 5 quadrados de distância na amostra
            for sy in range(1, 6):
                dst_pts = np.array([
                    [0, 0],
                    [sx, 0],
                    [0, sy],
                    [sx, sy]
                ], dtype=np.float32)
                
                try:
                    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
                except:
                    continue
                
                # 3. Projetar e contar inliers
                warped_pts = cv2.perspectiveTransform(pts_reais, H).reshape(-1, 2)
                nearest_int = np.round(warped_pts)
                dist = np.linalg.norm(warped_pts - nearest_int, axis=1)
                
                # Tolerância de 0.15 unidades de grid
                inliers_count = np.sum(dist < 0.15)
                
                if inliers_count > melhor_inliers_count:
                    melhor_inliers_count = inliers_count
                    melhor_H = H
    
    if melhor_H is not None:
        print(f"RANSAC Grid: Melhor modelo encontrado com {melhor_inliers_count} inliers.")
        
        # Recuperar extensão do grid
        warped_pts = cv2.perspectiveTransform(pts_reais, melhor_H).reshape(-1, 2)
        nearest_int = np.round(warped_pts).astype(int)
        dist = np.linalg.norm(warped_pts - nearest_int, axis=1)
        
        inliers_mask = dist < 0.15
        valid_coords = nearest_int[inliers_mask]
        
        if len(valid_coords) == 0:
            return None, [], None
            
        min_u, max_u = np.min(valid_coords[:, 0]), np.max(valid_coords[:, 0])
        min_v, max_v = np.min(valid_coords[:, 1]), np.max(valid_coords[:, 1])
        
        # Projetar de volta os cantos extremos encontrados
        try:
            H_inv = np.linalg.inv(melhor_H)
        except np.linalg.LinAlgError:
            print("Aviso: Matriz singular encontrada no RANSAC. Tentando inversão robusta...")
            # Tentar usar SVD ou pseudo-inversa se necessário, ou apenas falhar graciosamente
            try:
                H_inv = np.linalg.pinv(melhor_H)
            except:
                return None, [], None
        
        # Definir cantos do grid detectado
        grid_corners_warped = np.array([
            [min_u, min_v],
            [max_u, min_v],
            [min_u, max_v],
            [max_u, max_v]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        grid_corners_img = cv2.perspectiveTransform(grid_corners_warped, H_inv).reshape(-1, 2)
        cantos_finais = ordenar_cantos(grid_corners_img)
        
        # Retornar inliers reais para visualização
        inliers_reais = pts_reais[inliers_mask].reshape(-1, 2).astype(int)
        
        return cantos_finais, inliers_reais, None # Grid object simplificado

    return None, [], None
