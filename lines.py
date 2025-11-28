import numpy as np

# --- Funções Auxiliares ---

def _distancia_angular_pi(theta1, theta2):
    """Calcula a distância angular entre dois ângulos na faixa [0, pi), considerando periodicidade de pi."""
    theta1 = np.asarray(theta1)
    theta2 = np.asarray(theta2)
    
    diff = np.abs(theta1 - theta2)
    return np.minimum(diff, np.pi - diff)

def _media_circular(thetas):
    """Calcula a média circular para ângulos em [0, pi) usando método do dobramento."""
    thetas = np.asarray(thetas)
    # Dobrar ângulos para [0, 2pi), calcular média, depois dividir por 2
    sin_sum = np.sum(np.sin(2 * thetas))
    cos_sum = np.sum(np.cos(2 * thetas))
    mean_angle = np.arctan2(sin_sum, cos_sum) / 2
    # Garantir resultado em [0, pi)
    if mean_angle < 0:
        mean_angle += np.pi
    return mean_angle

# --- 1. Classificação com K-Means ---

def classificar_linhas(lines, max_iter=10):
    """
    Separa as linhas em dois grupos (Horizontais e Verticais) usando K-Means nos ângulos.
    """
    if lines is None or len(lines) < 2:
        return [], []

    lines_data = np.array([line[0] for line in lines])
    angles = lines_data[:, 1]
    
    # Inicialização robusta: usar mediana e ortogonal
    sorted_angles = np.sort(angles)
    c1 = sorted_angles[len(sorted_angles)//4]  # 25º percentil
    c2 = (c1 + np.pi/2) % np.pi

    for _ in range(max_iter):
        d1 = _distancia_angular_pi(angles, c1)
        d2 = _distancia_angular_pi(angles, c2)
        
        mask_g1 = d1 <= d2
        
        g1_angles = angles[mask_g1]
        g2_angles = angles[~mask_g1]

        if len(g1_angles) > 0:
            c1 = _media_circular(g1_angles)
        if len(g2_angles) > 0:
            c2 = _media_circular(g2_angles)
            
    # Separar linhas
    g1_lines = [lines[i] for i in range(len(lines)) if mask_g1[i]]
    g2_lines = [lines[i] for i in range(len(lines)) if not mask_g1[i]]
    
    return g1_lines, g2_lines

# --- 2. Agrupamento de Linhas Similares ---

def _ponto_medio_linha(rho, theta, img_shape=(1000, 1000)):
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

def _distancia_ponto_linha(ponto, rho, theta):
    """
    Calcula a distância perpendicular de um ponto (x, y) até a linha.
    Equação da linha: x*cos(theta) + y*sin(theta) = rho
    Distância = |x*cos(theta) + y*sin(theta) - rho|
    """
    x, y = ponto
    return abs(x * np.cos(theta) + y * np.sin(theta) - rho)

def filtrar_linhas_similares(linhas, threshold_dist=10, threshold_theta=np.pi/36, img_shape=(1000, 1000)):
    """
    Agrupa linhas similares e retorna representante de cada grupo.
    Usa distância perpendicular do ponto médio para comparar linhas.
    Retorna lista de tuplas (rho, theta) para compatibilidade com ransac_selecionar_linhas.
    """
    if len(linhas) == 0:
        return []
    
    # Extrair dados e calcular pontos médios
    lines_data = []
    for l in linhas:
        rho, theta = l[0][0], l[0][1]
        pm = _ponto_medio_linha(rho, theta, img_shape)
        lines_data.append((rho, theta, pm))
    
    # Ordenar por projeção do ponto médio (para ter ordem espacial consistente)
    # Usamos x*cos(theta_medio) + y*sin(theta_medio) como critério de ordenação
    theta_medio = _media_circular([l[1] for l in lines_data])
    
    def projecao_ordenacao(item):
        rho, theta, pm = item
        return pm[0] * np.cos(theta_medio) + pm[1] * np.sin(theta_medio)
    
    lines_data.sort(key=projecao_ordenacao)
    
    merged = []
    current_group = [lines_data[0]]
    
    for i in range(1, len(lines_data)):
        rho, theta, pm = lines_data[i]
        
        # Comparar com a PRIMEIRA linha do grupo (evita drift)
        first_rho, first_theta, first_pm = current_group[0]
        
        # Distância perpendicular do ponto médio atual até a primeira linha do grupo
        diff_dist = _distancia_ponto_linha(pm, first_rho, first_theta)
        diff_theta = _distancia_angular_pi(theta, first_theta)
        
        if diff_dist < threshold_dist and diff_theta < threshold_theta:
            current_group.append((rho, theta, pm))
        else:
            # Consolidar grupo: usar primeiro item
            first_rho, first_theta, _ = current_group[0]
            merged.append((first_rho, first_theta))
            current_group = [(rho, theta, pm)]
            
    # Último grupo
    if current_group:
        first_rho, first_theta, _ = current_group[0]
        merged.append((first_rho, first_theta))
    
    return merged

# --- 3. Pipeline Principal ---

def processar_linhas(lines, w, h):
    """
    Pipeline completo de processamento de linhas.
    Retorna listas de tuplas (rho, theta).
    """
    print(f"Total de linhas Hough: {len(lines)}")
    
    # 1. Classificar em H e V
    g1, g2 = classificar_linhas(lines)
    
    if len(g1) == 0 or len(g2) == 0:
        print("Erro: Não foi possível separar em H e V")
        return [], []
    
    # Identificar qual é H e qual é V
    # H tem theta próximo de pi/2 (90°), V tem theta próximo de 0 ou pi
    ang1 = _media_circular([l[0][1] for l in g1])
    ang2 = _media_circular([l[0][1] for l in g2])
    
    d1_to_h = _distancia_angular_pi(ang1, np.pi/2)
    d2_to_h = _distancia_angular_pi(ang2, np.pi/2)
    
    if d1_to_h < d2_to_h:
        linhas_h_raw = g1
        linhas_v_raw = g2
        theta_h_medio = ang1
        theta_v_medio = ang2
    else:
        linhas_h_raw = g2
        linhas_v_raw = g1
        theta_h_medio = ang2
        theta_v_medio = ang1
    
    print(f"Clusters: H={len(linhas_h_raw)} (θ={np.degrees(theta_h_medio):.1f}°), V={len(linhas_v_raw)} (θ={np.degrees(theta_v_medio):.1f}°)")
    
    # 2. Filtragem Angular (remover outliers > 20°)
    tolerancia_angular = np.pi / 9  # 20 graus
    
    linhas_h_filtradas = [l for l in linhas_h_raw 
                          if _distancia_angular_pi(l[0][1], theta_h_medio) < tolerancia_angular]
    linhas_v_filtradas = [l for l in linhas_v_raw 
                          if _distancia_angular_pi(l[0][1], theta_v_medio) < tolerancia_angular]
    
    print(f"Filtro angular (±20°): H {len(linhas_h_raw)}->{len(linhas_h_filtradas)}, V {len(linhas_v_raw)}->{len(linhas_v_filtradas)}")
        
    # 3. Agrupar linhas similares
    img_shape = (h, w)
    linhas_h = filtrar_linhas_similares(linhas_h_filtradas, threshold_dist=10, threshold_theta=np.pi/18, img_shape=img_shape)
    linhas_v = filtrar_linhas_similares(linhas_v_filtradas, threshold_dist=10, threshold_theta=np.pi/18, img_shape=img_shape)
    
    print(f"Agrupamento: H {len(linhas_h_filtradas)}->{len(linhas_h)}, V {len(linhas_v_filtradas)}->{len(linhas_v)}")
    
    return linhas_h, linhas_v