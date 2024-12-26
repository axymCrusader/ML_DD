import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import product
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def plot_comparison(true_values: np.ndarray, predicted_values: np.ndarray, 
                   title: str = 'Сравнение истинных и предсказанных значений') -> None:
    """
    Строит график сравнения истинных и предсказанных значений.

    Args:
        true_values: Массив истинных значений
        predicted_values: Массив предсказанных значений
        title: Заголовок графика

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label='Истинные значения')
    plt.plot(predicted_values, color='red', label='Предсказнные значения')
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()


def load_and_prepare_data(file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Загружает данные из Excel файла и подготавливает их для обучения модели.
    
    Включает:
    - Загрузку данных
    - Масштабирование признаков
    - Генерацию полиномиальных признаков второго порядка
    https://scikit-learn.ru/stable/modules/preprocessing.html
    
    Args:
        file_path: Путь к Excel файлу с данными

    Returns:
        tuple: (признаки в виде DataFrame, целевые значения в виде Series)
    """
    data = pd.read_excel(file_path)
    target_values = data.Type
    feature_values = data.drop(columns=['X', 'Y', 'Fi', 'Type'])
    
    scaler = StandardScaler()
    feature_values_scaled = scaler.fit_transform(feature_values)
    
    poly_features = []
    for i in range(feature_values_scaled.shape[1]):
        for j in range(i, feature_values_scaled.shape[1]):
            poly_features.append(feature_values_scaled[:, i] * feature_values_scaled[:, j])
    
    poly_features = np.column_stack(poly_features)
    
    final_features = np.hstack([feature_values_scaled, poly_features])
    
    original_columns = feature_values.columns
    poly_columns = [f'poly_{i}_{j}' for i in range(len(original_columns)) 
                   for j in range(i, len(original_columns))]
    
    final_features_df = pd.DataFrame(
        final_features,
        columns=list(original_columns) + poly_columns
    )
    
    return final_features_df, target_values


def create_neural_network(hidden_layers: tuple[int, ...] = (10, 10, 5),
                         activation: str = 'tanh',
                         solver: str = 'adam',
                         max_iter: int = 1000,
                         random_state: int = 42) -> MLPClassifier:
    """
    Создает и настраивает модель нейронной сети с заданными параметрами.

    Args:
        hidden_layers: Кортеж с количеством нейронов в каждом скрытом слое
        activation: Функция активации ('relu', 'tanh', 'logistic', 'identity')
        solver: Метод оптимизации ('adam', 'sgd', 'lbfgs')
        max_iter: Максимальное количество итераций
        random_state: Seed для воспроизводимости результатов

    Returns:
        MLPClassifier: Настроенная модель нейронной сети
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        learning_rate='adaptive',
        alpha=0.0001,
        batch_size='auto',
        learning_rate_init=0.001,
        power_t=0.5,
        momentum=0.9,
        nesterovs_momentum=True,
        tol=1e-5,
    )


def search_best_parameters(X: pd.DataFrame, y: pd.Series) -> tuple[dict, dict]:
    """
    Выполняет поиск оптимальных гиперпараметров модели.

    Перебирает различные комбинации параметров и оценивает их эффективность
    с помощью кросс-валидации. Сохраняет результаты в CSV файл.

    Args:
        X: DataFrame с признаками
        y: Series с целевыми значениями

    Returns:
        tuple: (лучшие параметры, метрики лучшей модели)
    """
    param_space = {
        'hidden_layers': [
            (10, 10, 5),
            (10, 10, 10),
            (10, 10, 15),
            (10, 10, 20),
            (100,),
            (200,),
            (300,),
            (100, 50),
            (200, 100),
            (300, 150),
            (100, 50, 25),
            (200, 100, 50),
            (300, 150, 75),
            (400, 200, 100),
            (200, 100, 50, 25),
            (300, 150, 75, 35),
            (400, 200, 100, 50),
            (500, 250, 125, 60)
        ],
        'activation': ['relu', 'tanh', 'logistic', 'identity'],
        'solver': ['adam', 'sgd', 'lbfgs'],
        'max_iter': [200, 500, 1000, 2000]
    }
    
    best_params = None
    best_metrics = {
        'f1': 0,
        'cv_mean': 0,
        'cv_std': float('inf')
    }
    
    results = []
    
    param_combinations = [dict(zip(param_space.keys(), v)) 
                         for v in product(*param_space.values())]
    
    total_combinations = len(param_combinations)
    print(f"Начало поиска параметров. Всего комбинаций: {total_combinations}")
    
    for idx, params in enumerate(param_combinations, 1):
        print(f"\rПроверка комбинации {idx}/{total_combinations}", end='')
        
        model = create_neural_network(
            hidden_layers=params['hidden_layers'],
            activation=params['activation'],
            solver=params['solver'],
            max_iter=params['max_iter']
        )
        
        try:
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='f1_macro')
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            model.fit(X, y)
            predictions = model.predict(X)
            
            f1 = f1_score(y, predictions, average='macro')
            accuracy = accuracy_score(y, predictions)
            
            result = {
                'hidden_layers': str(params['hidden_layers']),
                'activation': params['activation'],
                'solver': params['solver'],
                'max_iter': params['max_iter'],
                
                'accuracy': accuracy,
                'f1_score': f1,
                
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_fold_1': cv_scores[0],
                'cv_fold_2': cv_scores[1],
                'cv_fold_3': cv_scores[2],
                
                'validation_fraction': model.validation_fraction,
                'n_iter_no_change': model.n_iter_no_change,
                'learning_rate': model.learning_rate,
                'alpha': model.alpha,
                'batch_size': str(model.batch_size),
                'learning_rate_init': model.learning_rate_init,
                'momentum': model.momentum,
                'nesterovs_momentum': model.nesterovs_momentum,
                'early_stopping': model.early_stopping,
                'tol': model.tol,
                
                'n_iter_': model.n_iter_,
                'n_layers_': model.n_layers_,
                'n_outputs_': model.n_outputs_,
                
                'timestamp': pd.Timestamp.now()
            }
            
            results.append(result)
            
            if (cv_mean > best_metrics['cv_mean'] and cv_std < 0.1) or \
               (cv_mean >= best_metrics['cv_mean'] - 0.01 and cv_std < best_metrics['cv_std']):
                best_params = params
                best_metrics = {
                    'f1': f1,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std
                }
                
        except Exception as e:
            print(f"\nОшибка при проверке параметров: {params}")
            print(f"Ошибка: {str(e)}")
            continue
    
    results_df = pd.DataFrame(results)
    
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f'parameter_search_results_{timestamp}.csv'
    results_df.to_csv(filename, index=False)
    
    print(f"\nРезультаты сохранены в файл: {filename}")
    
    print("\nТоп-5 лучших конфигураций:")
    top_5 = results_df.head()
    for _, row in top_5.iterrows():
        print(f"\nКонфигурация:")
        print(f"hidden_layers: {row['hidden_layers']}")
        print(f"activation: {row['activation']}")
        print(f"solver: {row['solver']}")
        print(f"max_iter: {row['max_iter']}")
        print(f"F1-мера: {row['f1_score']:.4f}")
        print(f"Точность: {row['accuracy']:.4f}")
        print(f"Средняя CV: {row['cv_mean']:.4f}")
        print(f"Стд. откл. CV: {row['cv_std']:.4f}")
    
    return best_params, best_metrics


def evaluate_model(model: MLPClassifier, X: pd.DataFrame, y: pd.Series, is_test: bool = True) -> np.ndarray:
    """
    Оценивает качество модели по метрикам.

    Args:
        model: Обученная модель нейронной сети
        X: DataFrame с признаками
        y: Series с целевыми значениями
        is_test: Флаг, указывающий является ли это тестовой выборкой

    Returns:
        np.ndarray: Массив предсказанных значений
    """
    predictions = model.predict(X)
    
    if is_test:
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='macro')
        
        print(f"\nМетрики на тестовой выборке:")
        print(f"Точность модели: {round(accuracy * 100)}%")
        print(f"F1-мера: {round(f1 * 100)}%")
    
    return predictions


def split_dataset(X: pd.DataFrame, y: pd.Series, test_size: int) -> tuple:
    """
    Разбивает датасет на обучающую и тестовую выборки.

    Args:
        X: DataFrame с признаками
        y: Series с целевыми значениями
        test_size: Количество строк для тестовой выборки

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if test_size >= len(X):
        raise ValueError("Размер тестовой выборки не может быть больше или равен размеру датасета")
    
    test_fraction = test_size / len(X)
    return train_test_split(X, y, test_size=test_fraction, random_state=42, stratify=y)


def main() -> None:
    """
    Основная функция программы.
    """
    file_path = 'datasets/Data_Set_C.xlsx'
    X, y = load_and_prepare_data(file_path)
    
    while True:
        try:
            test_size = int(input(f"Введите размер тестовой выборки (доступно строк: {len(X)}): "))
            X_train, X_test, y_train, y_test = split_dataset(X, y, test_size)
            break
        except ValueError as e:
            print(f"Ошибка: {e}")
            continue

    model = create_neural_network(
        hidden_layers=(500, 250, 125, 60),
        activation='identity',
        solver='lbfgs',
        max_iter=200
    )
    
    model.fit(X_train, y_train)
    
    test_predictions = evaluate_model(model, X_test, y_test, is_test=True)
    plot_comparison(y_test, test_predictions, title='Сравнение на тестовой выборке')


if __name__ == '__main__':
    main()