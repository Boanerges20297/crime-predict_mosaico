# Figuras do artigo

Esta pasta concentra figuras exportadas a partir dos artefatos do projeto para apoiar a redacao do artigo.

## Subpastas
- `cobertura_95`: graficos sobre selecao dos bairros que compoem 95% da massa de eventos.
- `dispersao`: comparacoes entre valores observados e previstos.
- `media_movel`: series suavizadas por media movel.
- `mediana`: series suavizadas por mediana movel.
- `series_temporais`: curvas brutas agregadas no tempo.
- `hexagonos`: diagnosticos do sweep de configuracoes espaciais.

## Como regenerar
Execute:

```powershell
.\.venv\Scripts\python.exe .\src\generate_article_figures.py
```
