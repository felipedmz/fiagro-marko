import time
import pandas as pd
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
import numpy as np

def calculate(file_path):
    df = pd.read_csv(file_path, sep=";", index_col=0)
    # calculando os retornos esperados e a matriz de covariância
    retornos_esperados = mean_historical_return(df)
    matriz_covariancia = CovarianceShrinkage(df).ledoit_wolf()
    # Regularização: adicionando um pequeno valor à diagonal da matriz de covariância
    delta = 1e-4
    matriz_covariancia_regularizada = matriz_covariancia + delta * np.eye(matriz_covariancia.shape[0])
    # otimizando para o máximo Sharpe Ratio
    ef = EfficientFrontier(retornos_esperados, matriz_covariancia_regularizada)
    pesos = ef.max_sharpe()
    #print('... pesos', pesos)
    pesos_limpos = ef.clean_weights()
    #print('... pesos_limpos', pesos)
    # convertendo os pesos para porcentagens e salvando em um DataFrame
    pesos_porcentagem = {k: v * 100 for k, v in pesos_limpos.items()}
    df_pesos = pd.DataFrame(list(pesos_porcentagem.items()), columns=['Fundo', 'Alocação (%)'])
    return df_pesos
#
def run(filename):
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(f'> fiagro.markowitz - started={now}')
    start_time = time.time()
    print(f'... starting markowitz optimized allocation')
    result = calculate(filename)
    print(f'... best allocation found', result)
    output_file_name = "alocacao_optima.csv"
    result.to_csv(output_file_name, index=False)
    print(f'... best allocation found::: saved in `{output_file_name}`')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'> end - elapsed time={elapsed_time}')
#
