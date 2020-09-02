import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# print("Hola que tal ?") cada vez que importe va a meter esta funcion

def get_linear_regression(filename, plot_data=False):
    df = pd.read_csv(filename)
    lr = LinearRegression()
    lr.fit(df['Peso'].values.reshape(-1,1), df['Altura'].values.reshape(-1,1))
    if plot_data:
        plt.scatter(df['Peso'].values, df['Altura'].values)
        pesos = np.linspace(20, 130, 100)
        y = lr.predict(pesos.reshape(-1,1))
        plt.plot(pesos,y,c= 'y')
    return lr.coef_[0][0], lr.intercept_[0]

if __name__ == '__main__':
    print("Esta corriendo como script") # als coss con __ en python estan reservadas, se define automaticamente cuando yo IMPOROT con el nombre
else:
    print("Se esta importando como modulo")
        
    