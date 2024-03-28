import sys
import time
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage


def calculate(file_path):
    # Lendo o arquivo CSV
    df = pd.read_csv(file_path, sep=";", quotechar='"', index_col=0)

    # Calculando os retornos esperados e a matriz de covariância
    retornos_esperados = mean_historical_return(df)
    matriz_covariancia = CovarianceShrinkage(df).ledoit_wolf()

    # Otimizando para o máximo Sharpe Ratio
    ef = EfficientFrontier(retornos_esperados, matriz_covariancia)
    pesos = ef.max_sharpe()
    pesos_limpos = ef.clean_weights()

    # Convertendo os pesos para porcentagens e salvando em um DataFrame
    pesos_porcentagem = {k: v * 100 for k, v in pesos_limpos.items()}
    df_pesos = pd.DataFrame(list(pesos_porcentagem.items()), columns=['Fundo', 'Alocação (%)'])

    # Salvando os pesos otimizados em um arquivo CSV
    df_pesos.to_csv("alocacao_optima.csv", index=False)
#
def main():
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(f'> fiagro.markowitz - started={now}')
    start_time = time.time()
    calculate(sys.argv[1])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'> end - elapsed time={elapsed_time}')
#
main()
