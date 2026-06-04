import os
import unicodedata

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENRICHED_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "processed", "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv"),
    os.path.join(BASE_DIR, "data", "raw", "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv"),
]
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "fortaleza_crimes.csv")


def resolve_enriched_path():
    for path in ENRICHED_CANDIDATES:
        if os.path.exists(path):
            return path

    searched = "\n - ".join(ENRICHED_CANDIDATES)
    raise FileNotFoundError(f"Arquivo enriquecido não encontrado. Caminhos verificados:\n - {searched}")


def normalize_text(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()
    if not text or text == "NULL":
        return np.nan

    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.split())


def normalize_bairro_column(series):
    return series.apply(normalize_text)


def main():
    print("Iniciando processamento dos dados...")
    enriched_path = resolve_enriched_path()

    print(f"Lendo dados de {enriched_path}...")
    df = pd.read_csv(enriched_path, sep=",", low_memory=False)

    if "bairro" not in df.columns:
        raise KeyError("A coluna 'bairro' não foi encontrada na base enriquecida.")

    print("Normalizando nomes de bairro...")
    df["bairro"] = normalize_bairro_column(df["bairro"])

    if "cidade" in df.columns:
        df["cidade"] = df["cidade"].astype(str).str.strip()

    print("Filtrando registros de Fortaleza/CE...")
    capital_ais = ["AIS 05", "AIS 06", "AIS 08", "AIS 16", "AIS 17", "AIS 18", "AIS 19", "AIS 20", "AIS 21", "AIS 22"]
    df_fort = df[
        (df["cidade"].str.upper() == "FORTALEZA")
        & (df["ais"].isin(capital_ais) | df["ais"].isnull())
    ].copy()

    rmf_bairros = [
        "MARECHAL RONDON",
        "INDUSTRIAL",
        "PARQUE LEBLON",
        "IPARANA",
        "PACAJUS",
        "ITAITINGA",
        "PACATUBA",
        "HORIZONTE",
        "PINDORETAMA",
        "PARQUE SOLEDADE",
        "PARQUE ALBANO",
        "TABAPU",
        "PARQUE DAS NACOES",
        "CARARU",
        "URUCUTUBA",
        "DIF III",
        "SENADOR POMPEU",
        "JUAZEIRO DO NORTE",
        "PIRES FERREIRA",
        "IBICUITINGA",
        "REDENCAO",
    ]
    df_fort = df_fort[df_fort["bairro"].notna()].copy()
    df_fort = df_fort[~df_fort["bairro"].isin(rmf_bairros)].copy()

    print(f"Encontrados {len(df_fort)} registros legítimos de Fortaleza.")

    df_fort["latitude"] = pd.to_numeric(df_fort["latitude"], errors="coerce")
    df_fort["longitude"] = pd.to_numeric(df_fort["longitude"], errors="coerce")
    df_fort["data"] = pd.to_datetime(df_fort["data"], errors="coerce")

    initial_len = len(df_fort)
    df_fort = df_fort.dropna(subset=["latitude", "longitude"])
    removed_coords = initial_len - len(df_fort)
    if removed_coords > 0:
        print(f"Removidos {removed_coords} registros com coordenadas inválidas.")

    print("Processando associação Bairro -> AIS...")

    non_null_ais = df_fort[df_fort["ais"].notnull()]
    bairro_ais_map = non_null_ais.groupby("bairro")["ais"].agg(lambda values: values.value_counts().index[0]).to_dict()

    bairro_ais_map["GENTILANDIA"] = "AIS 05"

    bairros_com_ais = df_fort[df_fort["bairro"].isin(bairro_ais_map.keys())]
    centroids = bairros_com_ais.groupby("bairro")[["latitude", "longitude"]].mean()
    centroids["ais"] = centroids.index.map(bairro_ais_map)

    def get_closest_ais(row):
        lat, lon = row["latitude"], row["longitude"]
        dists = np.sqrt((centroids["latitude"] - lat) ** 2 + (centroids["longitude"] - lon) ** 2)
        closest_bairro = dists.idxmin()
        return centroids.loc[closest_bairro, "ais"]

    null_mask = df_fort["ais"].isnull()
    print(f"Total de registros com AIS nula inicialmente: {null_mask.sum()}")

    df_fort.loc[null_mask, "ais"] = df_fort.loc[null_mask, "bairro"].map(bairro_ais_map)

    still_null_mask = df_fort["ais"].isnull()
    still_null_count = still_null_mask.sum()
    if still_null_count > 0:
        print(f"Aplicando vizinho mais próximo para {still_null_count} registros de bairros novos...")
        closest_ais_values = df_fort[still_null_mask].apply(get_closest_ais, axis=1)
        df_fort.loc[still_null_mask, "ais"] = closest_ais_values

    print(f"Total de registros com AIS nula após tratamento: {df_fort['ais'].isnull().sum()}")

    print(f"Salvando dados processados em {OUTPUT_PATH}...")
    df_fort.to_csv(OUTPUT_PATH, index=False)
    print("Processamento concluído com sucesso!")


if __name__ == "__main__":
    main()
