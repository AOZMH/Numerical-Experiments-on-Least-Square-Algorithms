# 最优化上机作业三

* 运行实验
选择`main.py`中的相应测试函数并
> python main.py

* `osborne_func.py`及`data_fit_function.py`
实现Osborne最小二乘数据拟合测试函数，包括每个残差函数的计算、r(x)计算以及J(x)的计算

* `line_search.py`
Fibonacci及GLL线搜索实现

* `gauss_newton.py`
Gauss-Newton法的实现，同时支持类似DFP方法的大剩余量方法

* `dogleg.py`
基于单折（Single）及多折（Double）Dogleg方法的信赖域方法

* `orthogonal_distance_regression.py`
ODR方法所需的函数实现，即包括误差项的r(x)、J(x)计算，同时实现了J(x)的高效计算方法

* `odr_algo.py`
基于Dogleg的信赖域方法求解ODR问题
