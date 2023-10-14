import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import math
import pylab
import random
value = 1000#количество случайных значений
n = 1#сколько раз случайное число будет выбрано(объем выборки)
l = 0.05#параметр распределения случайной величины

#функция определения теоретической плотности
def eps_pdf(l, x):
    return (l * math.exp(-l * x))

#функция для определения выборочной плотности
def sample_expon(value, l, n):
    mu = 1 / l
    list_x = []
    for i in reversed(range(value)):#обратная последовательность
        summa = 0
        for j in reversed(range(n)):
            summa += mu ** 2 * random.expovariate(mu)
        list_x.append(summa / n)
    return list_x

# функция для формирования начальной таблицы,состоящей из дисперсии и мат.ожидания
def frame_start(l):
    frame = pd.DataFrame({
        'Распределение': 'Исходное',
        'Мат. ожидание': 1 / l,
        'Дисперсия': (math.sqrt(1 / l ** 2)) ** 2,
         # 'Стандарт. откл-е': math.sqrt(1 / l ** 2),
        'Мат. ожидание (откл-е)': 'NaN',
        'Мат. ожидание (откл-е, %)': 'NaN',
        'Дисперсия (откл-е)': 'NaN',
        'Дисперсия (откл-е, %)': 'NaN',
        # 'Стандарт. откл-е (откл-е)': 'NaN',
        # 'Стандарт. откл-е (откл-е, %)': 'NaN'
    }, index=[0])
    return frame

# функция для дополнения таблицы новой строкой I
def frame_conte(n, list_x):
    new_line = {
        'Распределение': 'Выборочное',
        'Объем выборки': n,
        'Мат. ожидание': round(np.mean(list_x), 3),
        'Дисперсия': round(np.var(list_x), 3)}
        # 'Стандарт. откл-е': round(np.std(list_x), 3)}
    return new_line

# напишем функцию для заполнения расчетных показателей первого порядка таблицы строки I
def frame_conte_dif_1(frame, n):
    i = frame.index[-1]
    frame['Мат. ожидание (откл-е)'].iloc[i] = (frame['Мат. ожидание'].iloc[i] - frame['Мат. ожидание'].iloc[0])
    frame['Дисперсия (откл-е)'].iloc[i] = (frame['Дисперсия'].iloc[i] - frame['Дисперсия'].iloc[0] / n)
    # frame['Стандарт. откл-е (откл-е)'].iloc[i] = (
    # frame['Стандарт. откл-е'].iloc[i] - frame['Мат. ожидание'].iloc[0] / (
    # (frame['Объем выборки'].iloc[i]) ** 0.5))

# напишем функцию для заполнения расчетных показателей второго порядка таблицы строки I
def frame_conte_dif_2(frame):
    i = frame.index[-1]
    frame['Мат. ожидание (откл-е, %)'].iloc[i] = round(100 * (
        frame['Мат. ожидание (откл-е)'].iloc[i] / frame['Мат. ожидание'].iloc[0]), 3)
    frame['Дисперсия (откл-е, %)'].iloc[i] = round(100 * (
        frame['Дисперсия (откл-е)'].iloc[i] / (frame['Дисперсия'].iloc[0] / frame['Объем выборки'].iloc[i])), 3)
    # frame['Стандарт. откл-е (откл-е, %)'].iloc[i] = round(float(100 * (
    # frame['Стандарт. откл-е (откл-е)'].iloc[i] / (
    # frame['Мат. ожидание'].iloc[0] / ((frame['Объем выборки'].iloc[i]) ** 0.5)))), 3)

 # функция для построения соответствий между выборочным и теоретическим плотностями распределения
def normal_pdf(x, mu, sigma):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

list_x = sample_expon(value, l, n)  #
list_xt = range(600)  # рандомный диапазон х значений
list_yt = [eps_pdf(l, x=i) for i in list_xt]  # диапазон y значений,сформированных в соответствии с экспоненциальным
# законом распределения
# коэффициент использования базовой последовательности - отношение длины требуемой последовательности к длине
# базовой последовательности
k1, k2 = sts.pearsonr(list_xt, list_yt)
print('Коэффициент использования базовой последовательности по оси х:')
print(k1)
print('Коэффициент использования базовой последовательности по оси y:')
print(k2)

# формирование выборочной плотности распределения
fig, axes = plt.subplots(figsize=(8, 4), dpi=80)  # макет колонок(столбцов) для гистограммы
plt.hist(list_x, bins='auto', color='orange', density=True, label='Выборочная плотность')  # гистограмма
# создаем нулевую таблицу (или обновляем таблицу) со значениями математического ожидания,дисперсии и среднего отклонения
frame = pd.DataFrame({'Create new frame': 'yes'}, index=[0])
frame = frame_start(l)
for i in range(n):
    frame = frame.append(frame_conte(n, list_x), ignore_index=True)
    frame_conte_dif_1(frame, n)
    frame_conte_dif_2(frame)
print(frame.loc[[0, 1], ['Распределение', 'Мат. ожидание', 'Дисперсия']])

axes.set_xlim([0, 150])  # масштаб оси х
plt.plot(list_xt, list_yt, label='Теоретическая плотность', color='crimson', linestyle='--', lw=4)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend((u'выборочное', u'теоретическое'))
plt.show()