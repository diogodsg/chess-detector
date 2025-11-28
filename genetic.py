import numpy as np

def ransac_selecionar_linhas(linhas, n_desejado=9, n_iter=2000, tolerancia=0.15, img_shape=(1000, 1000)):
    """
    Seleciona n_desejado linhas paralelas e equidistantes usando RANSAC.
    Usa o ponto médio das linhas na imagem para calcular distâncias.
    
    Em cada iteração:
    1. Amostra 2 linhas aleatórias para definir um espaçamento base
    2. Encontra todas as linhas que se encaixam nesse espaçamento (±tolerancia)
    3. Mantém o conjunto com mais inliers
    """
    if len(linhas) < n_desejado:
        return linhas
    
    # Normalizar linhas para ter theta no range [-pi/4, 3*pi/4]
    linhas_norm = []
    for rho, theta in linhas:
        if theta > 3 * np.pi / 4:
            linhas_norm.append((-rho, theta - np.pi))
        else:
            linhas_norm.append((rho, theta))
    
    # Calcular theta médio e filtrar outliers angulares
    thetas = [l[1] for l in linhas_norm]
    theta_medio = np.median(thetas)
    theta_std = np.std(thetas)
    
    # Filtrar linhas com theta muito diferente (tolerância adaptativa)
    tolerancia_theta = max(np.pi/36, theta_std * 2)  # mínimo 5 graus
    linhas_paralelas = [l for l in linhas_norm if abs(l[1] - theta_medio) < tolerancia_theta]
    
    if len(linhas_paralelas) < n_desejado:
        print(f"Aviso: Apenas {len(linhas_paralelas)} linhas paralelas, usando todas disponíveis")
        linhas_paralelas = linhas_norm
    
    # Calcular a direção perpendicular média (normal às linhas)
    theta_ref = np.median([l[1] for l in linhas_paralelas])
    
    # Função para calcular o ponto médio de uma linha na imagem
    def ponto_medio_linha(rho, theta, img_shape):
        """
        Calcula o ponto médio da linha dentro dos limites da imagem.
        Equação da linha Hough: x*cos(theta) + y*sin(theta) = rho
        """
        h, w = img_shape[:2]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Pontos de interseção com as bordas da imagem
        pontos = []
        
        # Interseção com x=0 (borda esquerda): y = rho / sin(theta)
        if abs(sin_t) > 1e-6:
            y = rho / sin_t
            if 0 <= y <= h:
                pontos.append((0, y))
        
        # Interseção com x=w (borda direita): y = (rho - w*cos(theta)) / sin(theta)
        if abs(sin_t) > 1e-6:
            y = (rho - w * cos_t) / sin_t
            if 0 <= y <= h:
                pontos.append((w, y))
        
        # Interseção com y=0 (borda superior): x = rho / cos(theta)
        if abs(cos_t) > 1e-6:
            x = rho / cos_t
            if 0 <= x <= w:
                pontos.append((x, 0))
        
        # Interseção com y=h (borda inferior): x = (rho - h*sin(theta)) / cos(theta)
        if abs(cos_t) > 1e-6:
            x = (rho - h * sin_t) / cos_t
            if 0 <= x <= w:
                pontos.append((x, h))
        
        if len(pontos) >= 2:
            # Ponto médio entre os dois pontos de interseção
            p1, p2 = pontos[0], pontos[1]
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        
        # Fallback: ponto mais próximo da origem na linha
        return (rho * cos_t, rho * sin_t)
    
    # Função para calcular a distância perpendicular de um ponto a uma linha
    def distancia_ponto_linha(ponto, rho, theta):
        """
        Calcula a distância perpendicular de um ponto (x, y) até a linha.
        Equação da linha: x*cos(theta) + y*sin(theta) = rho
        Distância = |x*cos(theta) + y*sin(theta) - rho|
        """
        x, y = ponto
        return abs(x * np.cos(theta) + y * np.sin(theta) - rho)
    
    # Calcular pontos médios de todas as linhas
    pontos_medios = []
    for rho, theta in linhas_paralelas:
        pm = ponto_medio_linha(rho, theta, img_shape)
        pontos_medios.append(pm)
    
    # Para ordenar as linhas, usamos a distância do ponto médio até uma linha de referência
    # A linha de referência passa pela origem com o theta médio (rho=0)
    # Distância = x*cos(theta_ref) + y*sin(theta_ref)
    def distancia_ordenacao(ponto):
        x, y = ponto
        return x * np.cos(theta_ref) + y * np.sin(theta_ref)
    
    # Criar lista de (linha, ponto_medio, distancia_ref) e ordenar
    linhas_com_info = []
    for i, (rho, theta) in enumerate(linhas_paralelas):
        pm = pontos_medios[i]
        dist_ref = distancia_ordenacao(pm)
        linhas_com_info.append(((rho, theta), pm, dist_ref))
    
    linhas_com_info.sort(key=lambda x: x[2])
    
    dist_min = linhas_com_info[0][2]
    dist_max = linhas_com_info[-1][2]
    range_total = dist_max - dist_min
    
    print(f"Range distância: {dist_min:.1f} a {dist_max:.1f} (delta={range_total:.1f})")
    print(f"Theta médio: {np.degrees(theta_ref):.1f}°")
    print(f"RANSAC: Iniciando {n_iter} iterações com {len(linhas_paralelas)} linhas candidatas...")
    
    melhor_inliers = []
    melhor_espacamento = 0
    
    for _ in range(n_iter):
        # 1. Amostrar 2 linhas aleatórias para definir espaçamento base
        idx = np.random.choice(len(linhas_com_info), 2, replace=False)
        idx.sort()
        
        linha1, pm1, dist1 = linhas_com_info[idx[0]]
        linha2, pm2, dist2 = linhas_com_info[idx[1]]
        
        # Calcular distância perpendicular do ponto médio da linha1 até a linha2
        espacamento_base = distancia_ponto_linha(pm1, linha2[0], linha2[1])
        
        if espacamento_base <= 0:
            continue
        
        # 2. Encontrar todas as linhas que formam uma sequência equidistante
        inliers = [(linha1, pm1, dist1)]
        
        # Buscar linhas para frente (ordenadas por distância de referência)
        for linha, pm, dist_ref in linhas_com_info:
            if dist_ref <= inliers[-1][2]:
                continue
            
            # Calcular distância perpendicular do ponto médio atual até esta linha
            dist = distancia_ponto_linha(inliers[-1][1], linha[0], linha[1])
            
            # Verificar se a distância está dentro da tolerância do espaçamento BASE
            # (não atualiza o espaçamento, compara sempre com o original)
            if espacamento_base > 0:
                ratio = dist / espacamento_base
                if (1 - tolerancia) <= ratio <= (1 + tolerancia):
                    inliers.append((linha, pm, dist_ref))
        
        # 3. Verificar se encontramos mais inliers que o melhor até agora
        if len(inliers) > len(melhor_inliers):
            melhor_inliers = inliers
            melhor_espacamento = espacamento_base
    
    print(f"RANSAC: Melhor modelo com {len(melhor_inliers)} inliers, espaçamento={melhor_espacamento:.1f}")
    
    # Extrair apenas as linhas dos inliers
    resultado = [item[0] for item in melhor_inliers]
    
    # Validar que as distâncias consecutivas respeitam a tolerância
    if len(melhor_inliers) >= 2:
        # Calcular distâncias perpendiculares consecutivas
        dists = []
        for i in range(len(melhor_inliers) - 1):
            linha_i, pm_i, _ = melhor_inliers[i]
            linha_j, pm_j, _ = melhor_inliers[i + 1]
            d = distancia_ponto_linha(pm_i, linha_j[0], linha_j[1])
            dists.append(d)
        
        mean_diff = np.mean(dists)
        std_diff = np.std(dists)
        cv = std_diff / mean_diff if mean_diff > 0 else 0
        print(f"Resultado: {len(resultado)} linhas, CV={cv:.4f}, espaçamento médio={mean_diff:.1f}")
    
    return resultado[:n_desejado]
