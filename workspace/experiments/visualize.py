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

def plot_format(title,xLabel,yLabel, size = 16):
    plt.title(title, fontsize=24)
    plt.xlabel(xLabel,fontsize=16)
    plt.ylabel(yLabel, fontsize=size)
    plt.tick_params(axis='both', which="major", labelsize=16)

if __name__ == "__main__":
    #获取数据
    num = 800
    gap = 1

    # data = read_image('progress.csv',num,gap)
    # plot_image(data,'blue','4x3 grid')

    # data1 = read_3image('400/2x2_400_1.csv','400/2x2_400_2.csv','400/2x2_400_3.csv',60)
    # data2= read_3image('400/1x3_400_1.csv','400/1x3_400_2.csv','400/1x3_400_3.csv', 60)
    # data3 = read_3image('400/2x3_400_1.csv','400/2x3_400_2.csv','400/2x3_400_3.csv',60)
    # data4 = read_3image('400/3x3_400_1.csv','400/3x3_400_2.csv','400/3x3_400_3.csv',60)

    # data1 = read_3image('600/2x2_600_1.csv','600/2x2_600_2.csv','600/2x2_600_3.csv',160)
    # data2= read_3image('600/1x3_600_1.csv','600/1x3_600_2.csv','600/1x3_600_3.csv',160)
    # data3 = read_3image('600/2x3_600_1.csv','600/2x3_600_2.csv','600/2x3_600_3.csv',160)
    # data4 = read_3image('600/3x3_600_1.csv','600/3x3_600_2.csv','600/3x3_600_3.csv',160)


    # data1 = read_3image('combine_action/6_1_1.csv','combine_action/6_1_2.csv','combine_action/6_1_3.csv',num)
    # data2= read_3image('combine_action/6_2_1.csv','combine_action/6_2_2.csv','combine_action/6_2_3.csv', num)
    # data3 = read_3image('combine_action/6_3_1.csv','combine_action/6_3_2.csv','combine_action/6_3_3.csv',num)
    # data4 = read_3image('combine_action/6_6_1.csv','combine_action/6_6_2.csv','combine_action/6_6_3.csv',num)

    # data1 = read_image('realign_data/normal.csv',175,gap)
    # data2 = read_image('realign_data/realign1.csv',175,gap)
    # data3 = read_image('realign_data/realign2.csv',175,gap)
    # data4 = read_image('realign_data/realign3.csv',175,gap)

    # data1 = read_image('compare_full/1x3.csv',800,4)
    # data2 = read_image('compare_full/full_1x3.csv',800,4)
    # data3 = read_image('compare_full/2x3.csv',1000,4)
    # data4 = read_image('compare_full/full_2x3.csv',1000,4)

    data1 = read_3image('Reward_function/no_value_reward_1.csv','Reward_function/no_value_reward_2.csv',
                                                'Reward_function/no_value_reward_3.csv',220)
    data2 = read_3image('Reward_function/value_reward_1.csv','Reward_function/value_reward_2.csv',
                                                'Reward_function/value_reward_3.csv',220)
    data3 = read_3image('Reward_function/no_value_critic_curve_1.csv','Reward_function/no_value_critic_curve_2.csv',
                                                'Reward_function/no_value_critic_curve_3.csv',150)
    data4 = read_3image('Reward_function/value_critic_curve_1.csv','Reward_function/value_critic_curve_2.csv',
                                                'Reward_function/value_critic_curve_3.csv',150)

##Compared with different DRL methods 1x3 600 flowrate
    # data1 = read_3image('compared_with_other_RL/1x3_600_1.csv','compared_with_other_RL/1x3_600_2.csv',
    # 'compared_with_other_RL/1x3_600_3.csv',1200)
    # data2 = read_3image('compared_with_other_RL/ppo_1.csv','compared_with_other_RL/ppo_2.csv',
    # 'compared_with_other_RL/ppo_3.csv',400)
    # data3 = read_3image('compared_with_other_RL/multi_ppo_1.csv','compared_with_other_RL/multi_ppo_2.csv',
    # 'compared_with_other_RL/multi_ppo_3.csv',400)
    # data4 = read_image('compared_with_other_RL/static.csv',1200)
    # data5 = read_image('compared_with_other_RL/actuated.csv',1200)


##Compared with different DRL methods 3x3 600 flowrate
    # data1 = read_3image('compared_with_other_RL/3x3_600_1.csv','compared_with_other_RL/3x3_600_2.csv',
    # 'compared_with_other_RL/3x3_600_3.csv',1200)
    # data2 = read_3image('compared_with_other_RL/3x3_single_1.csv','compared_with_other_RL/3x3_single_2.csv',
    # 'compared_with_other_RL/3x3_single_3.csv',400)
    # data3 = read_3image('compared_with_other_RL/3x3_multi_1.csv','compared_with_other_RL/3x3_multi_2.csv',
    # 'compared_with_other_RL/3x3_multi_3.csv',400)
    # data4 = read_image('compared_with_other_RL/3x3_static.csv',1200)
    # data5 = read_image('compared_with_other_RL/3x3_actuated.csv',1200)





    #绘制图形
    fig = plt.figure(dpi=128, figsize=(10,6))

##compared with different DRL methods
    # plot_3image(data1,'blue','ours',1)
    # plot_3image(data2,'red','Single-PPO',3,True)
    # plot_3image(data3,'black','Multi-PPO',3,True)
    # plot_image(data4,'green','Fixed-time')
    # plot_image(data5,'brown','Actuated')


##添加网格
    plt.grid() 
##400 inflow rate
    # plot_3image(data1,'blue','2x2 grid')
    # plot_3image(data2,'red','1x3 grid')
    # plot_3image(data3,'black','2x3 grid')
    # plot_3image(data4,'green','3x3 grid')
    # plot_format('','Episodes','Rewards')
    # plt.legend(['2x2 grid','1x3 grid','2x3 grid','3x3 grid'],loc = 4,fontsize = 20)

##combine actions
    # plot_3image(data1,'blue','one signal at a time')
    # plot_3image(data2,'red','two signals at a time')
    # plot_3image(data3,'black','three signals at a time')
    # plot_3image(data4,'green','six signals at a time')
    # plot_format('','Episodes','Rewards')
    # plt.legend(['one signal at a time','two signals at a time','three signals at a time','six signals at a time'],loc = 4,fontsize = 20)

##different action sequence 
    # plot_image(data1,'blue','sequence 1')
    # plot_image(data2,'red','sequence 2')
    # plot_image(data3,'black','sequence 3')
    # plot_image(data4,'green','sequence 4')
    # plot_format('','Episodes','Rewards')
    # plt.legend(['sequence 1','sequence 2','sequence 3','sequence 4'],loc = 4,fontsize = 20)
    

##compare with FCN
    # plot_image(data1,'blue','LTSM-agent')
    # plot_image(data2,'red','FCN-agent')
    # plot_image(data3,'blue','LTSM-agent')
    # plot_image(data4,'red','FCN-agent')
    # plot_format('','Episodes','Rewards')
    # plt.legend(['LSTM-agent','FCN-agent'],loc = 4,fontsize = 20)

##reward-to-go
    # plot_3image(data1,'blue','reward-to-go')
    # plot_3image(data2,'red','$G_t^k$-ours')
    # plot_format('','Episodes','Rewards')
    # plt.legend(['reward-to-go','our $G_t^k$'],loc = 4,fontsize = 20)

    plot_3image(data3,'blue','normal reward function')
    plot_3image(data4,'red','our reward function')
    plot_format('','Update steps','$\mathcal{L}_{critic}$', size=20)
    plt.legend(['reward-to-go','our $G_t^k$'],loc = 1,fontsize = 20)

##compared with different DRL methods
    # plot_format('1x3 grid under 600 flowrate','Episodes','Rewards')
    # plt.legend(['ours','Single-PPO','Multi-PPO','Fixed-time','Actuated'],loc = 4,fontsize = 15)


    plt.show()
