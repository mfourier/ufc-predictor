import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def random_ufc_data(test_size=0.2, random_state=42):
    np.random.seed(random_state)
    num_samples = 1000
    num_features = 10  # Ejemplo: atributos por peleador (golpes/minuto, tasa de KO, etc.)

    # Datos aleatorios para dos peleadores
    X_A = np.random.rand(num_samples, num_features)
    X_B = np.random.rand(num_samples, num_features)

    # Diferencia entre atributos (e.g., altura_A - altura_B)
    X = X_A - X_B

    # Etiquetas aleatorias (0: gana Fighter A, 1: gana Fighter B)
    y = np.random.randint(0, 2, num_samples)

    # Convertir a DataFrame
    columns = [f'feature_{i+1}' for i in range(num_features)]
    df = pd.DataFrame(X, columns=columns)
    df['winner'] = y

    # División en train y test
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['winner']), df['winner'],
                                                        test_size=test_size, random_state=random_state)
    
    # Normalización
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convertir de nuevo a DataFrame
    df_train = pd.DataFrame(X_train, columns=columns)
    df_test = pd.DataFrame(X_test, columns=columns)
    df_train['winner'] = y_train.values
    df_test['winner'] = y_test.values

    return df_train, df_test
