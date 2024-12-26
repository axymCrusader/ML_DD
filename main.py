import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score


def plot_ANN(result_rows, predict_ANN):
    plt.plot(result_rows, label='True Values')
    plt.plot(predict_ANN, color='red', label='Predicted Values')
    plt.title('График значений результирующего столбца и модели')
    plt.grid()
    plt.legend()
    plt.show()


def plot_result_rows(result_rows):
    plt.plot(result_rows)
    plt.title('График значений результирующего столбца')
    plt.ylabel('Значение')
    plt.grid()
    plt.show()


def get_xlsx_data(file_path):
    return pd.read_excel(file_path)


def main():
    file_path = 'datasets/Data_Set_(A+B).xlsx'
    data = get_xlsx_data(file_path)
    result_rows = data.Type
    input_rows = data.drop(columns=['X', 'Y', 'Fi', 'Type'])

    hidden_layer_size_param_first = 10
    hidden_layer_size_param_second = 10
    hidden_layer_size_param_third = 5

    activation_method = 'tanh'

    example_activation = {'identity', 'logistic', 'tanh', 'relu'}

    solver_method = 'adam'

    example_solver = {'lbfgs', 'sgd', 'adam'}

    _max_iter = 200

    _random_state = 42

    ANN = MLPClassifier(hidden_layer_sizes=(hidden_layer_size_param_first, hidden_layer_size_param_second,
                                            hidden_layer_size_param_third), activation=activation_method,
                        solver=solver_method, max_iter=_max_iter, random_state=_random_state)

    ANN.fit(input_rows, result_rows)
    predict_ANN = ANN.predict(input_rows)

    # plot_ANN(result_rows, predict_ANN)

    accuracy_data = accuracy_score(result_rows, predict_ANN)
    print(f"Процент угадываний: {round(accuracy_data * 100)}%")

    f1_data = f1_score(result_rows, predict_ANN)
    print(f"f1_score: {round(f1_data * 100)}%")


    data_cross_val_scores = cross_val_score(ANN, input_rows, result_rows, cv=3)
    for index, value in enumerate(data_cross_val_scores):
        print(f"Перекрестный критерий ошибки {index + 1}: {round(value * 100)}%")



if __name__ == '__main__':
    main()