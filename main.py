import cv2
import numpy as np
import sys
import os
from utils import desenhar_reta, calcular_intersecao, auto_canny, detecta_circulos, tira_circulos_fantasma
from lines import processar_linhas
from ransac import ransac_encontrar_tabuleiro
from genetic import ransac_selecionar_linhas

# blur forte para estimar fundo (tabuleiro / iluminação)
BACKGROUND_SIGMA = 35
# blur leve antes do Hough/Canny
BLUR_HOUGH_KERNEL = 7
BLUR_HOUGH_SIGMA = 1.5
         

def detecta_pecas(img_homo, nome_base):
    # equalização local (ajuda peças pretas e sombras)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_homo)


    # estima fundo com blur forte e divide -> “achatando” tabuleiro/iluminação
    bg = cv2.GaussianBlur(img_eq, (0, 0), BACKGROUND_SIGMA)
    norm = cv2.divide(img_eq, bg, scale=128)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # blur leve antes do Canny/Hough
    norm_blur = cv2.GaussianBlur(norm, (BLUR_HOUGH_KERNEL, BLUR_HOUGH_KERNEL), BLUR_HOUGH_SIGMA)

    # canny com limiares automáticos
    edges, low, high = auto_canny(norm_blur,
                                proporcao_fortes=0.05,
                                low_high_ratio=0.4,
                                sobel_ksize=3)

    print(f"Limiar Canny baixo = {low}, alto = {high}")

    # hough
    circulos_detectados = detecta_circulos(norm_blur, high)

    # elimina fantasmas
    circulos = tira_circulos_fantasma(circulos_detectados)

    # desenha resultado
    img_color = cv2.cvtColor(img_homo, cv2.COLOR_GRAY2BGR)
    for (x, y, r) in circulos:
        cv2.circle(img_color, (x, y), r, (0, 255, 0), 2)
        #cv2.circle(img_color, (x, y), 2, (0, 0, 255), 3)

    print("Peças detectadas:", len(circulos))
    cv2.imwrite(f"resultados/{nome_base}_pecas_detectadas.jpg", img_color)
    cv2.imwrite(f"resultados/{nome_base}Canny.jpg", edges)
   
    # cv2.imshow("01 - Original Gray", img_homo)
    # cv2.imshow("02 - Equalizada (CLAHE)", img_eq)
    # cv2.imshow("03 - Normalizada", norm)
    # cv2.imshow("04 - Canny (auto)", edges)
    # cv2.imshow("05 - Peças detectadas", img_color)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
  

def detectar_tabuleiro_xadrez(imagem_path):
    img = cv2.imread(imagem_path)
    if img is None:
        print(f"Erro ao carregar '{imagem_path}'")
        return
    
    # Extrair nome base da imagem para salvar resultados
    nome_base = os.path.splitext(os.path.basename(imagem_path))[0]
    
    # Criar diretório de resultados se não existir
    os.makedirs('resultados', exist_ok=True)
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Ajuste de parâmetros para detectar mais bordas
    edges = cv2.Canny(blur, 30, 150, apertureSize=3)
    # Hough com threshold baixo para detectar mais linhas, depois limitar às mais fortes
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
    
    # Limitar às 100 linhas mais fortes (HoughLines já retorna ordenado por votos)
    MAX_LINHAS = 50
    if lines is not None and len(lines) > MAX_LINHAS:
        lines = lines[:MAX_LINHAS]
    
    img_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img_lines = img.copy()
    img_hough = img.copy()

    if lines is not None:
        print(f"Retas detectadas: {len(lines)}")
        
        # Desenhar todas as linhas do Hough
        for line in lines:
            rho, theta = line[0]
            desenhar_reta(img_hough, rho, theta, (0, 255, 0))
        
        cv2.imwrite(f'resultados/{nome_base}_hough.jpg', img_hough)
        print(f"Salvo: {nome_base}_hough.jpg (todas as linhas detectadas)")
        
        # Processar e classificar linhas
        linhas_horizontais, linhas_verticais = processar_linhas(lines, w, h)
        print(f"Retas agrupadas - H: {len(linhas_horizontais)}, V: {len(linhas_verticais)})")
        
        # Desenhar linhas após agrupamento (antes do RANSAC)
        img_antes_ransac = img.copy()
        for rho, theta in linhas_horizontais:
            desenhar_reta(img_antes_ransac, rho, theta, (0, 0, 255),6)
        for rho, theta in linhas_verticais:
            desenhar_reta(img_antes_ransac, rho, theta, (255, 0, 0),6)
        
        cv2.imwrite(f'resultados/{nome_base}_antes_ransac.jpg', img_antes_ransac)
        print(f"Salvo: {nome_base}_antes_ransac.jpg (linhas agrupadas H/V)")

        # Selecionar as 9 melhores linhas de cada direção para formar grid 8x8
        N_LINHAS_GRID = 9
        
        if len(linhas_horizontais) >= N_LINHAS_GRID and len(linhas_verticais) >= N_LINHAS_GRID:
            print(f"Iniciando seleção de {N_LINHAS_GRID} linhas por segmentação...")
            
            # Tentar com tolerância crescente até encontrar 9 linhas
            best_h = []
            best_v = []
            
            # Tolerâncias crescentes: de 5% até 80%
            tolerancias = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]
            
            for tol in tolerancias:
                if len(best_h) < N_LINHAS_GRID:
                    best_h = ransac_selecionar_linhas(linhas_horizontais, N_LINHAS_GRID, img_shape=img.shape, tolerancia=tol)
                    print(f"  H: tolerância {tol:.2f} -> {len(best_h)} linhas")
                
                if len(best_v) < N_LINHAS_GRID:
                    best_v = ransac_selecionar_linhas(linhas_verticais, N_LINHAS_GRID, img_shape=img.shape, tolerancia=tol)
                    print(f"  V: tolerância {tol:.2f} -> {len(best_v)} linhas")
                
                if len(best_h) >= N_LINHAS_GRID and len(best_v) >= N_LINHAS_GRID:
                    break
            
            # Se ainda não temos 9 linhas, usar todas as disponíveis
            if len(best_h) < N_LINHAS_GRID:
                print(f"  Aviso: Não foi possível encontrar {N_LINHAS_GRID} linhas H, usando {len(best_h)}")
            if len(best_v) < N_LINHAS_GRID:
                print(f"  Aviso: Não foi possível encontrar {N_LINHAS_GRID} linhas V, usando {len(best_v)}")
            
            if best_h and best_v:
                print(f"RANSAC selecionou {len(best_h)}H e {len(best_v)}V linhas.")
                
                # Validar ortogonalidade
                theta_h_medio = np.median([theta for _, theta in best_h])
                theta_v_medio = np.median([theta for _, theta in best_v])
                angulo_diff = abs(theta_h_medio - theta_v_medio)
                ortogonalidade_erro = abs(angulo_diff - np.pi/2)
                print(f"Ortogonalidade: {np.degrees(ortogonalidade_erro):.2f}° de erro")
                
                linhas_horizontais = best_h
                linhas_verticais = best_v
                
                # Desenhar linhas selecionadas pelo RANSAC
                img_ransac = img.copy()
                for rho, theta in linhas_horizontais:
                    desenhar_reta(img_ransac, rho, theta, (0, 0, 255), thickness=2)
                for rho, theta in linhas_verticais:
                    desenhar_reta(img_ransac, rho, theta, (255, 0, 0), thickness=2)
                
                cv2.imwrite(f'resultados/{nome_base}_ransac_selecionadas.jpg', img_ransac)
                print(f"Salvo: {nome_base}_ransac_selecionadas.jpg (9H + 9V selecionadas)")
        else:
            print(f"Aviso: Insuficientes linhas (H:{len(linhas_horizontais)}, V:{len(linhas_verticais)})")
        
        # Desenhar linhas selecionadas
        for rho, theta in linhas_horizontais:
            desenhar_reta(img_lines, rho, theta, (0, 0, 255))
        for rho, theta in linhas_verticais:
            desenhar_reta(img_lines, rho, theta, (255, 0, 0))
        
        # Calcular interseções
        intersecoes = []
        for rho_h, theta_h in linhas_horizontais:
            for rho_v, theta_v in linhas_verticais:
                pt = calcular_intersecao(rho_h, theta_h, rho_v, theta_v)
                if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                    intersecoes.append(pt)
        
        print(f"Interseções: {len(intersecoes)}")
        
        # RANSAC para encontrar tabuleiro
        if len(linhas_horizontais) >= 2 and len(linhas_verticais) >= 2:
            # Se temos exatamente 9 linhas H e V, usar diretamente os cantos extremos
            if len(linhas_horizontais) == 9 and len(linhas_verticais) == 9:
                # Calcular os 4 cantos do grid 8x8 usando a primeira e última linha de cada direção
                h_primeira = linhas_horizontais[0]
                h_ultima = linhas_horizontais[-1]
                v_primeira = linhas_verticais[0]
                v_ultima = linhas_verticais[-1]
                
                # Cantos são as interseções das linhas extremas
                canto_tl = calcular_intersecao(h_primeira[0], h_primeira[1], v_primeira[0], v_primeira[1])
                canto_tr = calcular_intersecao(h_primeira[0], h_primeira[1], v_ultima[0], v_ultima[1])
                canto_bl = calcular_intersecao(h_ultima[0], h_ultima[1], v_primeira[0], v_primeira[1])
                canto_br = calcular_intersecao(h_ultima[0], h_ultima[1], v_ultima[0], v_ultima[1])
                
                if all([canto_tl, canto_tr, canto_bl, canto_br]):
                    cantos = [canto_tl, canto_tr, canto_bl, canto_br]
                    inliers = intersecoes  # Todas as interseções são inliers
                    print(f"Grid 8x8 definido diretamente pelas 9 linhas selecionadas")
                else:
                    cantos, inliers, _ = ransac_encontrar_tabuleiro(linhas_horizontais, linhas_verticais, intersecoes, n_iter=5000)
            else:
                cantos, inliers, _ = ransac_encontrar_tabuleiro(linhas_horizontais, linhas_verticais, intersecoes, n_iter=5000)
            
            if cantos is not None:
                # Destacar inliers
                for x, y in inliers:
                    cv2.circle(img_lines, (x, y), 8, (255, 0, 255), 2)
                
                tam = 800
                src = np.float32(cantos)
                dst = np.float32([[0, 0], [tam, 0], [0, tam], [tam, tam]])
                matriz = cv2.getPerspectiveTransform(src, dst)
                img_homo = cv2.warpPerspective(img, matriz, (tam, tam))
                
                # Desenhar cantos encontrados pelo RANSAC
                for i, pt in enumerate(cantos):
                    pt_int = (int(pt[0]), int(pt[1]))
                    cv2.circle(img_lines, pt_int, 12, (0, 255, 255), -1)
                    cv2.putText(img_lines, str(i+1), pt_int, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # Desenhar grid no resultado
                img_grid = img_homo.copy()
                casa = tam // 8
                for i in range(9):
                    pos = i * casa
                    cv2.line(img_grid, (pos, 0), (pos, tam), (0, 255, 0), 2)
                    cv2.line(img_grid, (0, pos), (tam, pos), (0, 255, 0), 2)
                
                # cv2.imshow('Homografia', img_homo)
                # cv2.imshow('Grid', img_grid)
                cv2.imwrite(f'resultados/{nome_base}_homografia.jpg', img_homo)
                cv2.imwrite(f'resultados/{nome_base}_grid.jpg', img_grid)
                
                # converter para cinza pois o detector de peças só aceita 1 canal
                img_homo_gray = cv2.cvtColor(img_homo, cv2.COLOR_BGR2GRAY)
                detecta_pecas(img_homo_gray, nome_base)


                # Criar imagem final combinada (2x2 grid)
                # Redimensionar todas para o mesmo tamanho
                tamanho_cell = 400
                img_orig_resized = cv2.resize(img, (tamanho_cell, tamanho_cell))
                img_hough_resized = cv2.resize(img_hough, (tamanho_cell, tamanho_cell))
                img_lines_resized = cv2.resize(img_lines, (tamanho_cell, tamanho_cell))
                img_grid_resized = cv2.resize(img_grid, (tamanho_cell, tamanho_cell))
                
                # Adicionar títulos
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_orig_resized, 'Original', (10, 30), font, 0.8, (0, 255, 0), 2)
                cv2.putText(img_hough_resized, 'Hough', (10, 30), font, 0.8, (0, 255, 0), 2)
                cv2.putText(img_lines_resized, 'Linhas', (10, 30), font, 0.8, (0, 255, 0), 2)
                cv2.putText(img_grid_resized, 'Grid 8x8', (10, 30), font, 0.8, (0, 255, 0), 2)
                
                # Montar grid 2x2
                linha_superior = np.hstack([img_orig_resized, img_hough_resized])
                linha_inferior = np.hstack([img_lines_resized, img_grid_resized])
                img_final = np.vstack([linha_superior, linha_inferior])
                
                cv2.imwrite(f'resultados/{nome_base}_final.jpg', img_final)
                print(f"Salvo: {nome_base}_final.jpg (imagem combinada)")
                
                print(f"\nRANSAC encontrou tabuleiro com {len(inliers)} pontos!")
            else:
                print("\nRANSAC não encontrou um tabuleiro válido.")
        
    else:
        print("Nenhuma reta detectada!")
    
    cv2.imwrite(f'resultados/{nome_base}_linhas.jpg', img_lines)


if __name__ == "__main__":
    import glob
    
    if len(sys.argv) > 1:
        # Processar imagem específica
        detectar_tabuleiro_xadrez(sys.argv[1])
    else:
        # Processar todas as imagens em img/
        extensoes = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        imagens = []
        for ext in extensoes:
            imagens.extend(glob.glob(f'img/{ext}'))
        
        if not imagens:
            print("Nenhuma imagem encontrada em img/")
        else:
            print(f"Encontradas {len(imagens)} imagens para processar\n")
            for i, img_path in enumerate(sorted(imagens)):
                print(f"\n{'='*60}")
                print(f"[{i+1}/{len(imagens)}] Processando: {img_path}")
                print('='*60)
                detectar_tabuleiro_xadrez(img_path)
