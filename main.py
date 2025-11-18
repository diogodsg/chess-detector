import cv2
import numpy as np
import sys


def desenhar_reta(img, rho, theta, cor=(0, 255, 0), espessura=2):
    """
    Desenha uma reta na imagem dado rho e theta da forma polar.
    """
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    
    # Calcular pontos extremos da linha
    x1 = int(x0 + 10000 * (-b))
    y1 = int(y0 + 10000 * (a))
    x2 = int(x0 - 10000 * (-b))
    y2 = int(y0 - 10000 * (a))
    
    cv2.line(img, (x1, y1), (x2, y2), cor, espessura)


def calcular_intersecao(rho1, theta1, rho2, theta2):
    """
    Calcula o ponto de interseção entre duas retas em forma polar.
    """
    a1 = np.cos(theta1)
    b1 = np.sin(theta1)
    a2 = np.cos(theta2)
    b2 = np.sin(theta2)
    
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    
    x = (b2 * rho1 - b1 * rho2) / det
    y = (a1 * rho2 - a2 * rho1) / det
    
    return (int(x), int(y))


def agrupar_retas_similares(retas, rho_threshold=30, theta_threshold=0.1):
    """
    Agrupa retas similares e retorna uma reta média para cada grupo.
    """
    if retas is None or len(retas) == 0:
        return []
    
    grupos = []
    usadas = set()
    
    for i, (rho1, theta1) in enumerate(retas):
        if i in usadas:
            continue
        
        grupo = [(rho1, theta1)]
        usadas.add(i)
        
        for j, (rho2, theta2) in enumerate(retas):
            if j in usadas:
                continue
            
            # Verificar se as retas são similares
            if (abs(rho1 - rho2) < rho_threshold and 
                abs(theta1 - theta2) < theta_threshold):
                grupo.append((rho2, theta2))
                usadas.add(j)
        
        # Calcular média do grupo
        rho_medio = np.mean([r for r, _ in grupo])
        theta_medio = np.mean([t for _, t in grupo])
        grupos.append((rho_medio, theta_medio))
    
    return grupos


def selecionar_linhas_extremas(linhas, num_linhas=9):
    """
    Seleciona linhas distribuídas uniformemente, incluindo as bordas.
    Para um tabuleiro 8x8, precisamos de 9 linhas (bordas + 7 internas).
    """
    if len(linhas) <= num_linhas:
        return sorted(linhas, key=lambda x: x[0])
    
    # Ordenar por rho (posição)
    linhas_ordenadas = sorted(linhas, key=lambda x: x[0])
    
    # Calcular espaçamento esperado entre linhas
    rho_values = [rho for rho, _ in linhas_ordenadas]
    rho_min, rho_max = min(rho_values), max(rho_values)
    
    # Agrupar linhas próximas e selecionar representantes
    linhas_selecionadas = []
    espaco_minimo = (rho_max - rho_min) / (num_linhas * 2)  # Espaçamento mínimo entre linhas
    
    for rho, theta in linhas_ordenadas:
        # Verificar se esta linha está longe o suficiente das já selecionadas
        adicionar = True
        for rho_sel, _ in linhas_selecionadas:
            if abs(rho - rho_sel) < espaco_minimo:
                adicionar = False
                break
        
        if adicionar:
            linhas_selecionadas.append((rho, theta))
    
    # Se temos mais que o esperado, selecionar distribuição uniforme
    if len(linhas_selecionadas) > num_linhas:
        indices = np.linspace(0, len(linhas_selecionadas) - 1, num_linhas, dtype=int)
        linhas_selecionadas = [linhas_selecionadas[i] for i in indices]
    
    return linhas_selecionadas


def detectar_tabuleiro_xadrez(imagem_path):
    """
    Detecta um tabuleiro de xadrez usando Transformada de Hough (retas) e homografia.
    
    Args:
        imagem_path: Caminho para a imagem do tabuleiro
    """
    # Carregar a imagem
    img = cv2.imread(imagem_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem '{imagem_path}'")
        return
    
    h, w = img.shape[:2]
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar blur para reduzir ruído
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordas usando Canny
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    
    # Aplicar Transformada de Hough para detectar RETAS (não segmentos)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    
    # Criar cópias da imagem para visualização
    img_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img_lines = img.copy()
    
    if lines is not None:
        print(f"Total de retas detectadas: {len(lines)}")
        
        # Converter para lista de tuplas (rho, theta)
        retas = [(line[0][0], line[0][1]) for line in lines]
        
        # Separar retas horizontais e verticais
        linhas_horizontais = []
        linhas_verticais = []
        
        for rho, theta in retas:
            # Converter theta para graus
            angulo_graus = theta * 180 / np.pi
            
            # Classificar como horizontal ou vertical
            # Horizontal: theta próximo de 0 ou 180 graus
            # Vertical: theta próximo de 90 graus
            if (angulo_graus < 20 or angulo_graus > 160):
                linhas_horizontais.append((rho, theta))
            elif (70 < angulo_graus < 110):
                linhas_verticais.append((rho, theta))
        
        # Agrupar retas similares
        linhas_horizontais = agrupar_retas_similares(linhas_horizontais, rho_threshold=20)
        linhas_verticais = agrupar_retas_similares(linhas_verticais, rho_threshold=20)
        
        print(f"Retas agrupadas:")
        print(f"  - Horizontais: {len(linhas_horizontais)}")
        print(f"  - Verticais: {len(linhas_verticais)}")
        
        # Selecionar 9 linhas em cada direção (bordas + internas)
        linhas_horizontais = selecionar_linhas_extremas(linhas_horizontais, num_linhas=9)
        linhas_verticais = selecionar_linhas_extremas(linhas_verticais, num_linhas=9)
        
        print(f"Linhas selecionadas (9 em cada direção):")
        print(f"  - Horizontais: {len(linhas_horizontais)}")
        print(f"  - Verticais: {len(linhas_verticais)}")
        
        # Desenhar retas na imagem
        for rho, theta in linhas_horizontais:
            desenhar_reta(img_lines, rho, theta, cor=(0, 0, 255), espessura=2)
        
        for rho, theta in linhas_verticais:
            desenhar_reta(img_lines, rho, theta, cor=(255, 0, 0), espessura=2)
        
        # Calcular interseções entre retas horizontais e verticais
        intersecoes = []
        for rho_h, theta_h in linhas_horizontais:
            for rho_v, theta_v in linhas_verticais:
                ponto = calcular_intersecao(rho_h, theta_h, rho_v, theta_v)
                if ponto is not None:
                    x, y = ponto
                    # Verificar se está dentro da imagem
                    if 0 <= x < w and 0 <= y < h:
                        intersecoes.append(ponto)
        
        print(f"Interseções encontradas: {len(intersecoes)}")
        
        # Desenhar interseções
        for x, y in intersecoes:
            cv2.circle(img_lines, (x, y), 5, (0, 255, 0), -1)
        
        # Se temos 81 interseções (9x9 grid completo), aplicar homografia
        if len(intersecoes) >= 4:
            # Encontrar os 4 cantos extremos (bordas do tabuleiro)
            top_left = min(intersecoes, key=lambda p: p[0] + p[1])
            top_right = max(intersecoes, key=lambda p: p[0] - p[1])
            bottom_left = min(intersecoes, key=lambda p: p[0] - p[1])
            bottom_right = max(intersecoes, key=lambda p: p[0] + p[1])
            
            # Os 4 cantos mapeiam diretamente para as bordas do tabuleiro
            cantos_origem = np.float32([top_left, top_right, bottom_left, bottom_right])
            
            # Definir tamanho do tabuleiro de destino (800x800)
            tamanho_tabuleiro = 800
            
            # Os cantos mapeiam para as bordas do quadrado
            cantos_destino = np.float32([
                [0, 0],                              # Top-left
                [tamanho_tabuleiro, 0],             # Top-right
                [0, tamanho_tabuleiro],             # Bottom-left
                [tamanho_tabuleiro, tamanho_tabuleiro]  # Bottom-right
            ])
            
            # Calcular matriz de homografia
            matriz_homografia = cv2.getPerspectiveTransform(cantos_origem, cantos_destino)
            
            # Aplicar transformação de perspectiva
            img_homografia = cv2.warpPerspective(img, matriz_homografia, 
                                                  (tamanho_tabuleiro, tamanho_tabuleiro))
            
            # Desenhar os 4 cantos do tabuleiro
            for i, ponto in enumerate([top_left, top_right, bottom_left, bottom_right]):
                cv2.circle(img_lines, ponto, 10, (255, 255, 0), -1)
                cv2.putText(img_lines, str(i+1), ponto, 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Desenhar grid completo no tabuleiro corrigido (8x8 = 9 linhas)
            img_grid = img_homografia.copy()
            tamanho_casa = tamanho_tabuleiro // 8
            
            for i in range(9):
                pos = i * tamanho_casa
                cv2.line(img_grid, (pos, 0), (pos, tamanho_tabuleiro), (0, 255, 0), 2)
                cv2.line(img_grid, (0, pos), (tamanho_tabuleiro, pos), (0, 255, 0), 2)
            
            # Adicionar labels das casas (a1-h8)
            img_labeled = img_grid.copy()
            colunas = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            for i in range(8):
                for j in range(8):
                    x = j * tamanho_casa + tamanho_casa // 2
                    y = i * tamanho_casa + tamanho_casa // 2
                    label = f"{colunas[j]}{8-i}"
                    cv2.putText(img_labeled, label, (x - 20, y + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            cv2.imshow('Tabuleiro Corrigido (Homografia)', img_homografia)
            cv2.imshow('Tabuleiro com Grid', img_grid)
            cv2.imshow('Tabuleiro com Labels', img_labeled)
            cv2.imwrite('resultado_homografia.jpg', img_homografia)
            cv2.imwrite('resultado_grid.jpg', img_grid)
            cv2.imwrite('resultado_labeled.jpg', img_labeled)
            print("\nHomografia aplicada com sucesso!")
            print("Tabuleiro mapeado para um quadrado 800x800")
            print("Imagens salvas: 'resultado_homografia.jpg', 'resultado_grid.jpg' e 'resultado_labeled.jpg'")
        
    else:
        print("Nenhuma reta detectada!")
    
    # Mostrar resultados
    cv2.imshow('Imagem Original', img)
    cv2.imshow('Bordas Detectadas (Canny)', img_edges)
    cv2.imshow('Retas Detectadas (Hough)', img_lines)
    
    print("\nPressione qualquer tecla para fechar as janelas...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Salvar resultados
    cv2.imwrite('resultado_bordas.jpg', img_edges)
    cv2.imwrite('resultado_linhas.jpg', img_lines)
    print("\nImagens salvas: 'resultado_bordas.jpg' e 'resultado_linhas.jpg'")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        imagem_path = sys.argv[1]
    else:
        # Caminho padrão - você pode mudar isso
        imagem_path = "img/chess.jpg"
        print(f"Usando imagem padrão: {imagem_path}")
        print("Uso: python main.py <caminho_da_imagem>\n")
    
    detectar_tabuleiro_xadrez(imagem_path)
