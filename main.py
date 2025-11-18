import cv2
import numpy as np
import sys

def detectar_tabuleiro_xadrez(imagem_path):
    """
    Detecta um tabuleiro de xadrez usando a Transformada de Hough para linhas.
    
    Args:
        imagem_path: Caminho para a imagem do tabuleiro
    """
    # Carregar a imagem
    img = cv2.imread(imagem_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem '{imagem_path}'")
        return
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar blur para reduzir ruído
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordas usando Canny
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    
    # Aplicar Transformada de Hough para detectar linhas
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=100, maxLineGap=10)
    
    # Criar cópias da imagem para visualização
    img_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img_lines = img.copy()
    
    if lines is not None:
        # Separar linhas horizontais e verticais
        linhas_horizontais = []
        linhas_verticais = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calcular ângulo da linha
            if x2 - x1 == 0:
                angulo = 90
            else:
                angulo = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
            
            # Classificar linhas como horizontais ou verticais
            if angulo < 45:  # Linha mais horizontal
                linhas_horizontais.append(line[0])
            else:  # Linha mais vertical
                linhas_verticais.append(line[0])
        
        # Desenhar linhas horizontais em vermelho
        for x1, y1, x2, y2 in linhas_horizontais:
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Desenhar linhas verticais em azul
        for x1, y1, x2, y2 in linhas_verticais:
            cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        print(f"Linhas detectadas:")
        print(f"  - Horizontais: {len(linhas_horizontais)}")
        print(f"  - Verticais: {len(linhas_verticais)}")
        print(f"  - Total: {len(lines)}")
        
        # Encontrar interseções para identificar o grid do tabuleiro
        intersecoes = []
        for h_line in linhas_horizontais:
            for v_line in linhas_verticais:
                # Calcular interseção entre linha horizontal e vertical
                x1h, y1h, x2h, y2h = h_line
                x1v, y1v, x2v, y2v = v_line
                
                # Simplificação: assumir linhas aproximadamente retas
                # Interseção aproximada
                xh_medio = (x1h + x2h) / 2
                yv_medio = (y1v + y2v) / 2
                
                # Adicionar ponto de interseção aproximado
                intersecoes.append((int(xh_medio), int(yv_medio)))
        
        # Desenhar interseções
        for x, y in intersecoes[:64]:  # Limitar a 64 pontos (8x8)
            cv2.circle(img_lines, (x, y), 5, (0, 255, 0), -1)
    
    else:
        print("Nenhuma linha detectada!")
    
    # Mostrar resultados
    cv2.imshow('Imagem Original', img)
    cv2.imshow('Bordas Detectadas (Canny)', img_edges)
    cv2.imshow('Linhas Detectadas (Hough)', img_lines)
    
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
        imagem_path = "img/203.jpg"
        print(f"Usando imagem padrão: {imagem_path}")
        print("Uso: python main.py <caminho_da_imagem>\n")
    
    detectar_tabuleiro_xadrez(imagem_path)
