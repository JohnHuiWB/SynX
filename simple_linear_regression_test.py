import graphlab  #模型制作以及数据处理模块
import matplotlib.pyplot as plt    #制图模块

sales = graphlab.SFrame('Datas/home_data.gl/')
print(sales)
graphlab.canvas.set_target('ipynb')  #可忽略，意味在jupyter中制图
sales.show(view = "Scatter Plot", x = "sqft_living", y = "price")   #只以简单的两个变量制图
train_data, test_data = sales.random_split(.8, seed = 0)         #以4：1分割训练集和测试集
sqft_model = graphlab.linear_regression.create(train_data, target = 'price', features = ['sqft_living'])
                                    #第一个简单模型
print(test_data['price'].mean())    #求测试集中的平均值
print(sqft_model.evaluate(test_data))  #评估所求模型
%matplotlib inline               #线性制图
plt.plot(test_data['sqft_living'], test_data['price'], '.',
         test_data['sqft_living'], sqft_model.predict(test_data),'-')
        #对测试集的回归制图与预测

sqft_model.get('coefficients')   #查看模型属性

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','floors',
              'zipcode']   #重新自定义新的特征属性

sales[my_features].show()     #查看统计图
sales.show(view = 'BoxWhisker Plot', x = 'zipcode', y = 'price')  #制作二维（双特征）柱状图
my_features_model = graphlab.linear_regression.create(train_data, target = 'price',
                                                     features = my_features)
       #自选多特征线性回归模型制作
print(my_features_model.evaluate(test_data))    #查看模型属性
house1 = sales[sales['id'] == '5309101200']     #查看单个房子
print(house1['price'])
sqft_model.predict(house1)
my_features_model.predict(house1)               #分别比较两个模型对房价的拟合情况
