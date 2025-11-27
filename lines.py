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

def filtrar_linhas_similares(linhas, threshold_rho=5, threshold_theta=np.pi/36):
    """
    Agrupa linhas similares e retorna representante de cada grupo.
    Retorna lista de tuplas (rho, theta) para compatibilidade com ransac_selecionar_linhas.
    """
    if len(linhas) == 0:
        return []
    
    # Extrair dados
    lines_data = [(l[0][0], l[0][1]) for l in linhas]
    
    # Ordenar por rho (mais importante para linhas paralelas)
    lines_data.sort(key=lambda x: x[0])
    
    merged = []
    current_group = [lines_data[0]]
    
    for i in range(1, len(lines_data)):
        rho, theta = lines_data[i]
        
        # Comparar com a PRIMEIRA linha do grupo (evita drift)
        first_rho, first_theta = current_group[0]
        
        diff_rho = abs(rho - first_rho)
        diff_theta = _distancia_angular_pi(theta, first_theta)
        
        if diff_rho < threshold_rho and diff_theta < threshold_theta:
            current_group.append((rho, theta))
        else:
            # Consolidar grupo: usar média
            avg_rho = np.mean([r for r, t in current_group])
            avg_theta = _media_circular([t for r, t in current_group])
            merged.append((avg_rho, avg_theta))
            current_group = [(rho, theta)]
            
    # Último grupo
    if current_group:
        avg_rho = np.mean([r for r, t in current_group])
        avg_theta = _media_circular([t for r, t in current_group])
        merged.append((avg_rho, avg_theta))
    
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
    linhas_h = filtrar_linhas_similares(linhas_h_filtradas, threshold_rho=5, threshold_theta=np.pi/18)
    linhas_v = filtrar_linhas_similares(linhas_v_filtradas, threshold_rho=5, threshold_theta=np.pi/18)
    
    print(f"Agrupamento: H {len(linhas_h_filtradas)}->{len(linhas_h)}, V {len(linhas_v_filtradas)}->{len(linhas_v)}")
    
    return linhas_h, linhas_v