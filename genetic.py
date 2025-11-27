import numpy as np

def ransac_selecionar_linhas(linhas, n_desejado=9, n_iter=2000):
    """
    Seleciona n_desejado linhas paralelas e equidistantes cobrindo todo o tabuleiro.
    Abordagem: dividir o range em n_desejado segmentos e buscar a melhor linha em cada segmento.
    """
    if len(linhas) < n_desejado:
        return linhas
    
    # Normalizar linhas
    linhas_norm = []
    for rho, theta in linhas:
        if theta > 3 * np.pi / 4:
            linhas_norm.append((-rho, theta - np.pi))
        else:
            linhas_norm.append((rho, theta))
    
    # Ordenar por rho
    linhas_norm.sort(key=lambda x: x[0])
    
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
    
    linhas_paralelas.sort(key=lambda x: x[0])
    
    rho_min = linhas_paralelas[0][0]
    rho_max = linhas_paralelas[-1][0]
    range_total = rho_max - rho_min
    
    print(f"Range: {rho_min:.1f} a {rho_max:.1f} (delta={range_total:.1f})")
    
    # Dividir o range em n_desejado segmentos
    segmento_size = range_total / n_desejado
    
    selecionadas = []
    
    for i in range(n_desejado):
        # Definir limites do segmento
        rho_start = rho_min + i * segmento_size
        rho_end = rho_min + (i + 1) * segmento_size
        rho_alvo = (rho_start + rho_end) / 2
        
        # Buscar linhas neste segmento
        candidatas_segmento = [l for l in linhas_paralelas 
                               if rho_start <= l[0] <= rho_end]
        
        if candidatas_segmento:
            # Pegar a linha mais próxima do centro do segmento
            melhor = min(candidatas_segmento, key=lambda x: abs(x[0] - rho_alvo))
            selecionadas.append(melhor)
        else:
            # Se não houver linha neste segmento, pegar a mais próxima do alvo
            melhor = min(linhas_paralelas, key=lambda x: abs(x[0] - rho_alvo))
            if melhor not in selecionadas:
                selecionadas.append(melhor)
            else:
                # Se já foi usada, pegar a segunda mais próxima
                candidatas_restantes = [l for l in linhas_paralelas if l not in selecionadas]
                if candidatas_restantes:
                    melhor = min(candidatas_restantes, key=lambda x: abs(x[0] - rho_alvo))
                    selecionadas.append(melhor)
    
    # Garantir que temos exatamente n_desejado linhas
    if len(selecionadas) < n_desejado:
        # Preencher com as linhas não usadas mais próximas dos gaps
        usadas = set(selecionadas)
        nao_usadas = [l for l in linhas_paralelas if l not in usadas]
        
        while len(selecionadas) < n_desejado and nao_usadas:
            # Encontrar o maior gap
            selecionadas.sort(key=lambda x: x[0])
            maior_gap = 0
            pos_gap = 0
            
            for i in range(len(selecionadas) - 1):
                gap = selecionadas[i+1][0] - selecionadas[i][0]
                if gap > maior_gap:
                    maior_gap = gap
                    pos_gap = i
            
            # Inserir linha no meio do gap
            rho_alvo = (selecionadas[pos_gap][0] + selecionadas[pos_gap+1][0]) / 2
            melhor = min(nao_usadas, key=lambda x: abs(x[0] - rho_alvo))
            selecionadas.append(melhor)
            nao_usadas.remove(melhor)
    
    selecionadas.sort(key=lambda x: x[0])
    
    # Validar equidistância
    if len(selecionadas) >= 2:
        rhos = [l[0] for l in selecionadas]
        diffs = np.diff(rhos)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        cv = std_diff / mean_diff if mean_diff > 0 else 0
        print(f"Selecionadas: {len(selecionadas)} linhas, CV={cv:.4f}, espaçamento médio={mean_diff:.1f}")
    
    return selecionadas[:n_desejado]
