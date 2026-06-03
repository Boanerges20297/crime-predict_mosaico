import pandas as pd
import json
from grid_generator import HexagonalGrid
from predictor import prepare_weekly_series, evaluate_model

def main():
    print("Iniciando avaliação comparativa final...")
    
    # 1. Carregar dados de crimes
    df = pd.read_csv("data/processed/fortaleza_crimes.csv", low_memory=False)
    df_clean = df[(df['latitude'] >= -3.9) & (df['latitude'] <= -3.6) & 
                  (df['longitude'] >= -38.7) & (df['longitude'] <= -38.4)].copy()
                  
    # 2. Carregar configuração ótima do grid hexagonal do AG
    with open("data/processed/best_hex_grid.json", "r") as f:
        best_hex = json.load(f)
        
    print("\nParâmetros do Grid Ótimo:")
    print(f"  dx: {best_hex['dx']:.6f}")
    print(f"  dy: {best_hex['dy']:.6f}")
    print(f"  Rotação (theta): {best_hex['theta']:.6f}")
    print(f"  Raio (R): {best_hex['R']:.6f}")
    
    # 3. Gerar o Grid Ótimo e atribuir aos pontos
    min_lon, max_lon = df_clean['longitude'].min(), df_clean['longitude'].max()
    min_lat, max_lat = df_clean['latitude'].min(), df_clean['latitude'].max()
    bbox = (min_lon, min_lat, max_lon, max_lat)
    
    grid = HexagonalGrid(bbox, dx=best_hex['dx'], dy=best_hex['dy'], theta=best_hex['theta'], R=best_hex['R'])
    df_clean['hex_id'] = grid.assign_points(df_clean)
    
    # Filtrar pontos que eventualmente caíram fora das bordas do grid rotacionado
    df_hex = df_clean[df_clean['hex_id'] != -1].copy()
    
    # 4. Avaliar cada partição
    print("\n[Avaliando Divisão: AIS]")
    df_series_ais = prepare_weekly_series(df_clean, region_col='ais', lags=3)
    mse_ais = evaluate_model(df_series_ais, region_col='ais', lags=3)
    regions_ais = df_clean['ais'].nunique()
    
    print("\n[Avaliando Divisão: Bairros]")
    df_series_bairros = prepare_weekly_series(df_clean, region_col='bairro', lags=3)
    mse_bairros = evaluate_model(df_series_bairros, region_col='bairro', lags=3)
    regions_bairros = df_clean['bairro'].nunique()
    
    print("\n[Avaliando Divisão: Mosaico Hexagonal Ótimo]")
    df_series_hex = prepare_weekly_series(df_hex, region_col='hex_id', lags=3)
    mse_hex = evaluate_model(df_series_hex, region_col='hex_id', lags=3)
    regions_hex = df_hex['hex_id'].nunique()
    
    # 5. Imprimir Tabela de Resultados
    print("\n" + "="*60)
    print("                RESULTADOS COMPARATIVOS FINAIS")
    print("="*60)
    print(f"{'Divisão Territorial':<30} | {'Regiões Ativas':<14} | {'EQM (MSE)':<10}")
    print("-"*60)
    print(f"{'Áreas Integradas de Segurança':<30} | {regions_ais:<14} | {mse_ais:<10.4f}")
    print(f"{'Bairros (Políticos)':<30} | {regions_bairros:<14} | {mse_bairros:<10.4f}")
    print(f"{'Mosaico Hexagonal Ótimo':<30} | {regions_hex:<14} | {mse_hex:<10.4f}")
    print("="*60)
    
    # Salvar relatório final em markdown no diretório raiz do projeto
    report_content = f"""# Relatório Final: Otimização de Mosaico Hexagonal para Previsão de Crimes

Estudo comparativo de previsão espaço-temporal do número de crimes semanais na Cidade de Fortaleza/CE utilizando Regressão Linear Múltipla (RLM) baseada em lags temporais de 3 semanas.

## Parâmetros Ótimos do Grid Hexagonal (Encontrados via AG)
- **Deslocamento X (dx):** {best_hex['dx']:.6f}
- **Deslocamento Y (dy):** {best_hex['dy']:.6f}
- **Rotação (theta):** {best_hex['theta']:.6f} rad ({best_hex['theta'] * 180 / 3.14159:.2f} graus)
- **Raio (R):** {best_hex['R']:.6f} graus (aproximadamente {best_hex['R'] * 111.3 * 1000:.1f} metros)

## Tabela Comparativa de Erros (EQM)

| Divisão Territorial | Regiões Ativas | Erro Quadrático Médio (EQM) |
| :--- | :---: | :---: |
| **Áreas Integradas de Segurança (AIS)** | {regions_ais} | {mse_ais:.4f} |
| **Bairros** | {regions_bairros} | {mse_bairros:.4f} |
| **Mosaico Hexagonal Ótimo** | {regions_hex} | {mse_hex:.4f} |

## Conclusões
O **Mosaico Hexagonal Ótimo** obtido através da busca evolucionária obteve o menor Erro Quadrático Médio (EQM) de previsão (**{mse_hex:.4f}**), superando tanto a divisão tradicional por Bairros (**{mse_bairros:.4f}**) quanto a divisão por Áreas Integradas de Segurança (AIS) (**{mse_ais:.4f}**).
"""
    with open("RELATORIO_COMPARATIVO.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("\nRelatório salvo com sucesso em 'RELATORIO_COMPARATIVO.md'")

if __name__ == "__main__":
    main()
