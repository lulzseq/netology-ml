# Задание 

Взять проект с [прошлого домашнего задания](https://github.com/lulzseq/netology-ml/blob/master/DS%20project%20management%20methodology.ipynb). Структурировать его код, отчет и файлы с данными на основе лекции.

# Цель

Предсказать оценку вина на основе доступных данных о нем.

# Описание

Есть два набора данных с красными и белыми вариантами португальского вина "Vinho Verde". Из-за проблем конфиденциальности и логистики доступны только физико-химические (входные) и сенсорные (выходные) переменные (например, нет данных о типах винограда, марке вина, цене продажи вина и т.п.).

Эти наборы данных можно рассматривать как задачи классификации, так и регрессии. Классы упорядочены и не сбалансированы (например, есть более нормальные вина, чем отличные или плохие). Алгоритмы обнаружения выбросов могут быть использованы для обнаружения нескольких отличных или плохих вин. В датасетах есть сомнения, что все входные переменные актуальны, поэтому следует это проверить. Два набора данных были объединены, и несколько значений были случайно удалены.

# Установка библиотек

* cycler==0.10.0
* joblib==0.15.1
* kiwisolver==1.2.0
* matplotlib==3.2.1
* numpy==1.18.5
* pandas==1.0.4
* pickle-mixin==1.0.2
* pyparsing==2.4.7
* python-dateutil==2.8.1
* pytz==2020.1
* scikit-learn==0.23.1
* scipy==1.4.1
* seaborn==0.10.1
* six==1.15.0
* threadpoolctl==2.1.0
* xgboost==1.1.1