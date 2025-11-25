import cv2
import numpy as np
import sys


def desenhar_reta(img, rho, theta, cor=(0, 255, 0)):
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    pts = [(int(x0 + 10000 * (-b)), int(y0 + 10000 * a)),
           (int(x0 - 10000 * (-b)), int(y0 - 10000 * a))]
    cv2.line(img, pts[0], pts[1], cor, 2)


def calcular_intersecao(rho1, theta1, rho2, theta2):
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    det = a1 * b2 - a2 * b1
    
    if abs(det) < 1e-10:
        return None
    
    x = (b2 * rho1 - b1 * rho2) / det
    y = (a1 * rho2 - a2 * rho1) / det
    return (int(x), int(y))


def agrupar_retas(retas, rho_thr=20, theta_thr=0.1):
    if not retas:
        return []
    
    grupos, usadas = [], set()
    
    for i, (rho1, theta1) in enumerate(retas):
        if i in usadas:
            continue
        
        grupo = [(rho1, theta1)]
        usadas.add(i)
        
        for j, (rho2, theta2) in enumerate(retas):
            if j not in usadas and abs(rho1 - rho2) < rho_thr and abs(theta1 - theta2) < theta_thr:
                grupo.append((rho2, theta2))
                usadas.add(j)
        
        grupos.append((np.mean([r for r, _ in grupo]), np.mean([t for _, t in grupo])))
    
    return grupos


def selecionar_linhas(linhas, n=9):
    if len(linhas) <= n:
        return sorted(linhas, key=lambda x: x[0])
    
    linhas = sorted(linhas, key=lambda x: x[0])
    rhos = [rho for rho, _ in linhas]
    espaco_min = (max(rhos) - min(rhos)) / (n * 2)
    
    selecionadas = []
    for rho, theta in linhas:
        if all(abs(rho - r) >= espaco_min for r, _ in selecionadas):
            selecionadas.append((rho, theta))
    
    if len(selecionadas) > n:
        indices = np.linspace(0, len(selecionadas) - 1, n, dtype=int)
        selecionadas = [selecionadas[i] for i in indices]
    
    return selecionadas


def detectar_tabuleiro_xadrez(imagem_path):
    img = cv2.imread(imagem_path)
    if img is None:
        print(f"Erro ao carregar '{imagem_path}'")
        return
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    
    img_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img_lines = img.copy()

    if lines is not None:
        print(f"Retas detectadas: {len(lines)}")
        
        # Separar horizontais e verticais
        linhas_horizontais, linhas_verticais = [], []
        for line in lines:
            rho, theta = line[0]
            angulo = theta * 180 / np.pi
            if angulo < 20 or angulo > 160:
                linhas_horizontais.append((rho, theta))
            elif 70 < angulo < 110:
                linhas_verticais.append((rho, theta))
        
        # Agrupar e selecionar retas
        linhas_horizontais = agrupar_retas(linhas_horizontais)
        linhas_verticais = agrupar_retas(linhas_verticais)
        
        print(f"Retas agrupadas - H: {len(linhas_horizontais)}, V: {len(linhas_verticais)}")
        
        linhas_horizontais = selecionar_linhas(linhas_horizontais, 9)
        linhas_verticais = selecionar_linhas(linhas_verticais, 9)
        
        print(f"Linhas selecionadas - H: {len(linhas_horizontais)}, V: {len(linhas_verticais)}")
        
        # Desenhar retas
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
        
        for x, y in intersecoes:
            cv2.circle(img_lines, (x, y), 5, (0, 255, 0), -1)
        
        # Aplicar homografia
        if len(intersecoes) >= 4:
            cantos = [
                min(intersecoes, key=lambda p: p[0] + p[1]),
                max(intersecoes, key=lambda p: p[0] - p[1]),
                min(intersecoes, key=lambda p: p[0] - p[1]),
                max(intersecoes, key=lambda p: p[0] + p[1])
            ]
            
            tam = 800
            src = np.float32(cantos)
            dst = np.float32([[0, 0], [tam, 0], [0, tam], [tam, tam]])
            matriz = cv2.getPerspectiveTransform(src, dst)
            img_homo = cv2.warpPerspective(img, matriz, (tam, tam))
            
            # Desenhar cantos
            for i, pt in enumerate(cantos):
                cv2.circle(img_lines, pt, 10, (255, 255, 0), -1)
                cv2.putText(img_lines, str(i+1), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
          
            
            cv2.imshow('Homografia', img_homo)
            cv2.imwrite('resultado_homografia.jpg', img_homo)
            print("\nHomografia aplicada! Imagens salvas.")
        
    else:
        print("Nenhuma reta detectada!")
    
    cv2.imshow('Original', img)
    cv2.imshow('Bordas', img_edges)
    cv2.imshow('Linhas', img_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('resultado_bordas.jpg', img_edges)
    cv2.imwrite('resultado_linhas.jpg', img_lines)


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "img/chess.jpg"
    if len(sys.argv) == 1:
        print(f"Usando: {img_path}\nUso: python main.py <imagem>\n")
    detectar_tabuleiro_xadrez(img_path)

# Afrouxar regra de agrupamento
# Usar ransac para detectar os 4 melhores pontos que batem com nosso resultado