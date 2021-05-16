#导入csv模块
import numpy as np
import csv
from matplotlib import pyplot as plt
import numpy.random

def read_image(filename,num,gap = 1):
    with open(filename) as f:
        #创建一个阅读器：将f传给csv.reader
        reader = csv.reader(f)
        #使用csv的next函数，将reader传给next，将返回文件的下一行
        header_row = next(reader)
        
        # for index, column_header in enumerate(header_row):
        #         print(index, column_header)

        highs =[]
        #遍历reader的余下的所有行（next读取了第一行，reader每次读取后将返回下一行）
        count = 0
        for row in reader:
            if (count < num)  & ((count % gap) == 0):
        #将字符串转换成数字
                high = float(row[2])
                highs.append(high)
            count += 1
    return highs

def plot_image(data,color,label):
    plt.plot(data,c=color,label = label,lw=1)


def read_3image(f1,f2,f3,num):
    f = [f1, f2, f3]
    img = [open(f[i]) for i in range(len(f))]
    #创建一个阅读器：将f传给csv.reader
    reader = [csv.reader(img[i]) for i in range(len(f))]
    header_row1 = next(reader[0])
    header_row2 = next(reader[1])
    header_row3 = next(reader[2])

    data = [[],[],[]]
    out = [[],[]]
    count = 0
    for row in reader[0]:
        if count < num:
            data[0].append(float(row[2]))
        count += 1
    count = 0
    for row in reader[1]:
        if count < num:
            data[1].append(float(row[2]))
        count += 1
    count = 0
    for row in reader[2]:
        if count < num:
            data[2].append(float(row[2]))
        count += 1
    print(num)
    print(np.array(data).shape)
    for i in range(num):
        # print(i)
        temp = sorted([data[0][i],data[1][i],data[2][i]])
        mean = (temp[0] + temp[1] +temp[2])/3
        std = ((temp[0] - mean) ** 2 + (temp[1] - mean) ** 2 + (temp[2] - mean) ** 2)/3 
        out[0].append(mean)
        out[1].append(std ** (1/2))
    return np.array(out)

def plot_3image(data,color,label,longer = 1,random=False):

    random_num = np.zeros([len(data[0])])
    if random == True:
        random_num = np.random.randint(3,6,size=len(data[0])) / 4

    plt.plot(np.array(range(len(data[0]))) * longer, data[0] - random_num,c=color,label = label,lw=1)
    plt.fill_between(np.array(range(len(data[0]))) * longer, data[0] - data[1]-random_num, 
    data[0] + data[1]-random_num, facecolor=color, alpha=0.3)

def paint_3img(f1,f2,f3,color,label):
    data = read_3image(f1,f2,f3)
    plot_3image(data,color,label)

def plot_format(title,xLabel,yLabel):
    plt.title(title, fontsize=24)
    plt.xlabel(xLabel,fontsize=16)
    plt.ylabel(yLabel, fontsize=16)
    plt.tick_params(axis='both', which="major", labelsize=16)

if __name__ == "__main__":
    #获取数据
    grid = "2x2"


##Compared with different DRL methods 3x3 600 flowrate
    data1 = read_3image('600_{}_ours_1.csv'.format(grid),'600_{}_ours_2.csv'.format(grid),
    '600_{}_ours_3.csv'.format(grid),200)
    data2 = read_3image('600_{}_single_1.csv'.format(grid),'600_{}_single_2.csv'.format(grid), 
    '600_{}_single_3.csv'.format(grid),200)
    data3 = read_3image('600_{}_multi_1.csv'.format(grid),'600_{}_multi_2.csv'.format(grid),
    '600_{}_multi_3.csv'.format(grid),200)
    data4 = read_image('600_{}_fix.csv'.format(grid),200)
    data5 = read_image('600_{}_actuated.csv'.format(grid),200)

    #绘制图形
    fig = plt.figure(dpi=128, figsize=(10,6))

##compared with different DRL methods
    plot_3image(data1,'blue','ours')
    plot_3image(data2,'red','Single-PPO')
    plot_3image(data3,'black','Multi-PPO')
    plot_image(data4,'green','Fixed-time')
    plot_image(data5,'brown','Actuated')

##compared with different DRL methods
    plot_format('{} grid under 600 flowrate'.format(grid),'Episodes','Rewards')
    plt.legend(['ours','Single-PPO','Multi-PPO','Fixed-time','Actuated'],loc = 4,fontsize = 15)


    plt.show()
