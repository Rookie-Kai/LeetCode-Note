## Exploratory Data Analysis 数据探索性分析

## EDA数据探索性分析

## 一、EDA目标

- 熟悉、了解数据集，验证所得数据集可以用于接下来的机器学习或深度学习使用
- 了解变量间的相互关系以及变量与预测值之间的关系
- 引导后续进行数据处理及特征工程的步骤，使得数据集的结构和特征集让接下来的预测问题更加可靠
- 完成对于数据的探索性分析，并进行总结

### 1、熟悉、了解数据集，验证所得数据集可以用于接下来的机器学习或深度学习使用

本目标使用python各数据科学及可视化库完成：
数据科学库——pandas、numpy、scipy
可视化库——matplotlib、seanbon；

实践步骤：

### a、载入数据，并简略观察（head()+shape() ）：

#### 1) 载入训练集和测试集；
```
Train_data = pd.read_csv('train.csv', sep=' ')
Test_data = pd.read_csv('testA.csv', sep=' ')
```

#### 2) 简略观察数据(head()+shape)
```
Train_data.head().append(Train_data.tail())
Train_data.shape #观察数据形状
Test_data.head().append(Test_data.tail()) #观察head+tail两部分数据叠加显示
```
### b、数据总览，(describe()熟悉数据统计量，info()熟悉数据类型）

#### 1) 通过describe()来熟悉数据的相关统计量

```
Train_data.describe()，Test_data.describe()
```

#### 2) 通过info()来熟悉数据类型

```
Train_data.info()
Test_data.info()
```

### c、判断数据缺失和异常
（查看每列存在nan的情况，异常值检测）

#### 1) 查看每列的存在nan情况
```
Train_data.isnull().sum()，Test_data.isnull().sum()

# nan可视化
missing = Train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)#sort_values 排序函数，如果inplace设置为False，输出的树状图仍为乱序
missing.plot.bar()
```

通过以上两句可以很直观的了解哪些列存在 “nan”, 并可以把nan的个数打印，主要的目的在于 nan存在的个数是否真的很大，如果很小一般选择填充，如果使用lgb等树模型可以直接空缺，让树自己去优化，但如果nan存在的过多、可以考虑删掉

----------------------------------------------------------
#### 参数

```
sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, by=None)
```

#### 参数说明
- axis：0按照行名排序；1按照列名排序
- level：默认None，否则按照给定的level顺序排列---貌似并不是，文档
- ascending：默认True升序排列；False降序排列
- inplace：默认False，否则排序之后的数据直接替换原来的数据框
- kind：默认quicksort，排序的方法
- na_position：缺失值默认排在最后{"first","last"}
- by：按照那一列数据进行排序，但是by参数貌似不建议使用

### 可视化看下缺省值

```
msno.matrix(Train_data.sample(250)) #矩阵式黑色填充数据，nan以空缺替代
msno.bar(Train_data.sample(1000))    #统计非nan的数据数并以柱形图展示
```

#### 2) 查看异常值检测

```
Train_data.info() #查看数据的类型
Train_data['notRepairedDamage'].value_counts() #查看数据中各个value的计数
#如果有非nan的值比如“-”需要怎么处理？可以直接替换成nan，因为很多模型对nan有直接的处理。
Train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
Train_data.isnull().sum() #使用.isnull统计各数据nan的数量

# 处理和判断空值的笔记： 
用pandas或numpy处理数据中的空值（np.isnan()/pd.isnull()）_Python_甘如荠-CSDN博客

# 对于使用describe观察之后的数据，如果觉得数据过于异常，可以使用value_counts()查看数据分布之后，发现严重偏斜的可以直接删掉。

del Train_data["seller"]
del Train_data["offerType"]
del Test_data["seller"]
del Test_data["offerType"]
```

### d、了解预测值的分布

（总体分布情况（这里需要熟悉各种概率统计分布（这里是约翰逊分布 ）；查看偏度skewness和峰度kurtosis）；查看预测值的具体频数）
```
Train_data['price']
Train_data['price'].value_counts()
```

#### 1) 总体分布概况（无界约翰逊分布等）

```
import scipy.stats as st
y = Train_data['price']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu) #绘制图像并拟合johnsonsu曲线 scipy.statsy用法参考笔记： python统计函数库scipy.stats的用法1/3_Python_潜水的飞鱼baby-CSDN博客
plt.figure(2); plt.title('Normal') #seanborn.distplot用法文档： seaborn.distplot-Seaborn 0.9 中文文档，入门教程： Python可视化 | Seaborn5分钟入门(一)——kdeplot和distplot - 简书
sns.distplot(y, kde=False, fit=st.norm) #绘制图像并拟合正态分布
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm) #绘制图像并拟合log正态
```

价格不服从正态分布，所以在进行回归之前，它必须进行转换。（很多框架模型都要求数据服从正态分布，另外过偏的数据特征相近很难提取）

log变化的用途参考笔记，log可以将偏态转换为正态 log变换：https://app.yinxiang.com/fx/ee907718-23d4-464e-addc-9731bdb5470f

#### 2) 查看skewness and kurtosis

```
sns.distplot(Train_data['price']);
print("Skewness: %f" % Train_data['price'].skew())
print("Kurtosis: %f" % Train_data['price'].kurt())
Train_data.skew(), Train_data.kurt() #查看数据的偏度和峰度
sns.distplot(Train_data.skew(),color='blue',axlabel ='Skewness') #对偏度绘图
sns.distplot(Train_data.kurt(),color='orange',axlabel ='Kurtness') #峰度绘图
```

数据的偏度和峰度的数学原理教程： 数据的偏度和峰度——df.skew()、df.kurt() - 喜欢吃面的Hush - 博客园

#### 3) 查看预测值的具体频数
```
plt.hist(Train_data['price'], orientation = 'vertical',histtype = 'bar', color ='red')
plt.show()
```

查看频度，如果某一区间的数据频数相对极低，那么考虑当做特殊值将其填充替代或者删掉。

```
# log变换 z之后的分布较均匀，可以进行log变换进行预测，这也是预测问题常用的trick

plt.hist(np.log(Train_data['price']), orientation = 'vertical',histtype = 'bar', color ='red')
plt.show()
```

### e、特征分为类别特征和数值特征，并对类别特征查看unique分布。
```
# 分离label即预测值
Y_train = Train_data['price']
如果没有直接label coding的数据，可以人为的定义区分：
# 数字特征
# numeric_features = Train_data.select_dtypes(include=[np.number])
# numeric_features.columns
# # 类型特征
# categorical_features = Train_data.select_dtypes(include=[np.object])
# categorical_features.columns
numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v..
categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDam..

# 特征nunique分布
for cat_fea in categorical_features:
    print(cat_fea + "的特征分布如下：")
    print("{}特征有个{}不同的值".format(cat_fea, Train_data[cat_fea].nunique()))
    print(Train_data[cat_fea].value_counts())
    （得到特征分布之后即可根据其特征进行分析）
```

### f、数字特征分析
(相关性分析；查看几个特征的偏度和峰值；每个数字特征的分布可视化；数字特征相互之间的关系可视化；多变量互相回归关系可视化）

数据相关性分析参考笔记：数据 相关性 分析

numeric_features.append('price') #数字特征中加入price

#### 1) 相关性分析
```
price_numeric = Train_data[numeric_features]
correlation = price_numeric.corr() #DataFrame.corr()用来判断数据集合间的相关性
print(correlation['price'].sort_values(ascending = False),'\n')
pandas的DataFrame.corr()用法详解： pandas相关系数-DataFrame.corr()参数详解_Python_walking_visitor的博客-CSDN博客

f , ax = plt.subplots(figsize = (7, 7)) #定义子图 subplot和subplots绘制子图_Python_lyzkks的博客-CSDN博客

plt.title('Correlation of Numeric Features with Price',y=1,size=16)

sns.heatmap(correlation,square = True, vmax=0.8) #绘制热力图，可以观察相关的数据

heatmap介绍：seaborn.heatmap参数介绍_Python_菜菜鸟的博客-CSDN博客
```

#### 2) 查看几个特征得 偏度和峰值
```
for col in numeric_features:
print('{:15}'.format(col),
'Skewness: {:05.2f}'.format(Train_data[col].skew()) ,
' ' ,
'Kurtosis: {:06.2f}'.format(Train_data[col].kurt())
)
```
#### 3) 每个数字特征得分布可视化
```
f = pd.melt(Train_data, value_vars=numeric_features) #将numberic_features进行转换（变成var+value列向量）
g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False) #建立方格
g = g.map(sns.distplot, "value") #绘制散点
FaceGrid和map搭配绘图讲解：Seaborn学习（一）------- 构建结构化多绘图网格（FacetGrid(）、map()）详解_Python_进击的菜鸟-CSDN博客
```

#### 4) 数字特征相互之间的关系可视化
```
sns.set()
columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5', 'v_2', 'v_6', 'v_1', 'v_14']
sns.pairplot(Train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()
```
#### 5) 多变量互相回归关系可视化
```
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2,
# ['v_12', 'v_8' , 'v_0', 'power', 'v_5', 'v_2', 'v_6', 'v_1', 'v_14']
v_12_scatter_plot = pd.concat([Y_train,Train_data['v_12']],axis = 1) 
#横向拼接Y_train和,Train_data['v_12']
sns.regplot(x='v_12',y = 'price', data = v_12_scatter_plot,scatter= True, fit_reg=True, ax=ax1)
v_8_scatter_plot = pd.concat([Y_train,Train_data['v_8']],axis = 1)
sns.regplot(x='v_8',y = 'price',data = v_8_scatter_plot,scatter= True, fit_reg=True, ax=ax2)
v_0_scatter_plot = pd.concat([Y_train,Train_data['v_0']],axis = 1)
sns.regplot(x='v_0',y = 'price',data = v_0_scatter_plot,scatter= True, fit_reg=True, ax=ax3)
power_scatter_plot = pd.concat([Y_train,Train_data['power']],axis = 1)
sns.regplot(x='power',y = 'price',data = power_scatter_plot,scatter= True, fit_reg=True, ax=ax4)
v_5_scatter_plot = pd.concat([Y_train,Train_data['v_5']],axis = 1)
sns.regplot(x='v_5',y = 'price',data = v_5_scatter_plot,scatter= True, fit_reg=True, ax=ax5)
v_2_scatter_plot = pd.concat([Y_train,Train_data['v_2']],axis = 1)
sns.regplot(x='v_2',y = 'price',data = v_2_scatter_plot,scatter= True, fit_reg=True, ax=ax6)
v_6_scatter_plot = pd.concat([Y_train,Train_data['v_6']],axis = 1)
sns.regplot(x='v_6',y = 'price',data = v_6_scatter_plot,scatter= True, fit_reg=True, ax=ax7)
v_1_scatter_plot = pd.concat([Y_train,Train_data['v_1']],axis = 1)
sns.regplot(x='v_1',y = 'price',data = v_1_scatter_plot,scatter= True, fit_reg=True, ax=ax8)
v_14_scatter_plot = pd.concat([Y_train,Train_data['v_14']],axis = 1)
sns.regplot(x='v_14',y = 'price',data = v_14_scatter_plot,scatter= True, fit_reg=True, ax=ax9)
v_13_scatter_plot = pd.concat([Y_train,Train_data['v_13']],axis = 1)
sns.regplot(x='v_13',y = 'price',data = v_13_scatter_plot,scatter= True, fit_reg=True, ax=ax10)
```

### h、类型特征分析
（unique分布；类别特征箱型图可视化；类别特征的小提琴图可视化；类别特征的圆柱形图可视化；类别特征的圆柱形图可视化类别；特征的每个类别频数可视化（count_plot））
#### 1) unique分布
```
for fea in categorical_features:
print(Train_data[fea].nunique()) #返回fea中的唯一值的个数
```
#### 2) 类别特征箱形图可视化
```
# 因为 name和 regionCode的类别太稀疏了，这里我们把不稀疏的几类画一下
categorical_features = ['model',
'brand',
'bodyType',
'fuelType',
'gearbox',
'notRepairedDamage']
for c in categorical_features:
Train_data[c] = Train_data[c].astype('category')
if Train_data[c].isnull().any():
Train_data[c] = Train_data[c].cat.add_categories(['MISSING'])
Train_data[c] = Train_data[c].fillna('MISSING')
def boxplot(x, y, **kwargs):
sns.boxplot(x=x, y=y)
x=plt.xticks(rotation=90)
f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "price")
```

#### 3) 类别特征的小提琴图可视化
```
catg_list = categorical_features
target = 'price'
for catg in catg_list :
sns.violinplot(x=catg, y=target, data=Train_data)
plt.show()
```
#### 4) 类别特征的柱形图可视化
```
def bar_plot(x, y, **kwargs):
sns.barplot(x=x, y=y)
x=plt.xticks(rotation=90)
f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(bar_plot, "value", "price")
```

#### 5) 类别特征的每个类别频数可视化(count_plot)
```
def count_plot(x, **kwargs):
sns.countplot(x=x)
x=plt.xticks(rotation=90)
f = pd.melt(Train_data, value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(count_plot, "value")
```

#### j、用pandas_profiling生成数据报告

用pandas_profiling生成一个较为全面的可视化和数据报告(较为简单、方便) 最终打开html文件即可
```
import pandas_profiling
pfr = pandas_profiling.ProfileReport(Train_data)
pfr.to_file("./example.html")
```

## 二、经验总结
所给出的EDA步骤为广为普遍的步骤，在实际的不管是工程还是比赛过程中，这只是最开始的一步，也是最基本
的一步。

接下来一般要结合模型的效果以及特征工程等来分析数据的实际建模情况，根据自己的一些理解，查阅文献，对实际问题做出判断和深入的理解。

最后不断进行EDA与数据处理和挖掘，来到达更好的数据结构和分布以及较为强势相关的特征

数据探索有利于我们发现数据的一些特性，数据之间的关联性，对于后续的特征构建是很有帮助的。

1. 对于数据的初步分析（直接查看数据，或.sum(), .mean()，.descirbe()等统计函数）可以从：样本数量，训练集数量，是否有时间特征，是否是时许问题，特征所表示的含义（非匿名特征），特征类型（字符类似，int，float，time），特征的缺失情况（注意缺失的在数据中的表现形式，有些是空的有些是”NAN”符号等），特征的均值方差情况。
2. 分析记录某些特征值缺失占比30%以上样本的缺失处理，有助于后续的模型验证和调节，分析特征应该是填充（填充方式是什么，均值填充，0填充，众数填充等），还是舍去，还是先做样本分类用不同的特征模型去预测。
3. 对于异常值做专门的分析，分析特征异常的label是否为异常值（或者偏离均值较远或者事特殊符号）,异常值是否应该剔除，还是用正常值填充，是记录异常，还是机器本身异常等。
4. 对于Label做专门的分析，分析标签的分布情况等。
5. 进步分析可以通过对特征作图，特征和label联合做图（统计图，离散图），直观了解特征的分布情况，通过这一步也可以发现数据之中的一些异常值等，通过箱型图分析一些特征值的偏离情况，对于特征和特征联合作图，对于特征和label联合作图，分析其中的一些关联性。


如何解决console输出数据被隐藏问题？
可以通过pandas的显示设置解决：
```
pd.set_option('display.max_columns', None) #显示完整的列
pd.set_option('display.max_rows', None) #显示完整的行

#可视化
missing = Train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace = True) #如果inplace设置为False，输出的树状图仍为乱序
missing.plot.bar()
```
----------------------------------------------------------
### 参数
```
sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, by=None)
```
#### 参数说明
- axis：0按照行名排序；1按照列名排序
- level：默认None，否则按照给定的level顺序排列---貌似并不是，文档
- ascending：默认True升序排列；False降序排列
- inplace：默认False，否则排序之后的数据直接替换原来的数据框
- kind：默认quicksort，排序的方法
- na_position：缺失值默认排在最后{"first","last"}
- by：按照那一列数据进行排序，但是by参数貌似不建议使用

```
rain_data['notRepairedDamage'].replace('-',np.nan,inplace=True)
```

pandas 中 inplace 参数在很多函数中都会有，它的作用是：是否在原对象基础上进行修改

inplace = True：不创建新的对象，直接对原始对象进行修改；

inplace = False：对数据进行修改，创建并返回新的对象承载其修改结果。

Train_data.skew(), Train_data.kurt()

--- 

### skew定义

偏度（skewness），是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。偏度(Skewness)亦称偏态、偏态系数。
表征概率分布密度曲线相对于平均值不对称程度的特征数。直观看来就是密度函数曲线尾部的相对长度。
定义上偏度是样本的三阶标准化矩：

### 方法

DataFrame.skew(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)
kurt是计算峰度的。