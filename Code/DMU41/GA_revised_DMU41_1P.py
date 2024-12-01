# -*- coding: utf-8 -*-
"""
#GA_revised_one_point
#DMU 41
#適應度值為 makespan

Created on Wed Jun 19 15:16:10 2024

@author: user
"""

import numpy as np
# import math


# ==== 參數設定(與問題相關) ====

NUM_JOB = 20            # 工件個數   # === Step 1. 設定成 20 個工件 ===
NUM_MACHINE = 15                     # === Step 1. 設定 15 個機台

# === Step 2. 設定 processing time 矩陣(row: job ID; col: 對應其加工機台所用的加工時間) ===
pTime = [
           [80,	47,	8,	4,	94,	195,	190,	165,	168,	26,	48,	75,	109,	175,	196],
[151,	182,	52,	144,	103,	138,	102,	109,	168,	52,	10,	138,	27,	149,	192],
[181,	151,	147,	45,	5,	43,	57,	164,	15,	116,	163,	69,	13,	106,	70],
[97,	180,	43,	186,	123,	58,	108,	145,	111,	182,	61,	60,	195,	100,	193],
[73,	195,	88,	22,	91,	168,	58,	30,	128,	126,	107,	150,	76,	147,	61],
[87,	31,	5,	22,	163,	125,	159,	3,	88,	110,	26,	19,	13,	175,	46],
[193,	172,	88,	37,	155,	94,	136,	150,	30,	45,	60,	124,	171,	176,	132],
[136,	2,	99,	81,	69,	130,	132,	165,	26,	11,	123,	7,	123,	170,	105],
[32,	124,	51,	172,	164,	82,	16,	194,	131,	135,	168,	141,	119,	5,	125],
[80,	188,	153,	38,	104,	6,	17,	193,	114,	7,	109,	123,	11,	88,	45],
[52,	28,	179,	200,	170,	128,	185,	17,	137,	77,	84,	135,	94,	59,	13],
[37,	196,	1,	180,	5,	101,	2,	153,	112,	142,	164,	175,	136,	45,	199],
[117,	193,	147,	44,	70,	33,	49,	46,	5,	128,	106,	150,	3,	134,	24],
[80,	26,	144,	106,	150,	188,	94,	185,	7,	68,	171,	23,	78,	151,	196],
[62,	57,	152,	68,	200,	86,	84,	189,	19,	19,	63,	17,	165,	78,	157],
[56,	111,	20,	99,	38,	152,	172,	67,	173,	145,	55,	199,	89,	140,	140],
[87,	87,	85,	7,	41,	46,	27,	30,	178,	137,	59,	66,	71,	63,	21],
[86,	14,	111,	97,	52,	71,	140,	21,	135,	40,	22,	123,	146,	131,	33],
[118,	177,	98,	7,	16,	181,	98,	37,	8,	71,	99,	158,	133,	71,	195],
[171,	2,	119,	189,	107,	21,	18,	90,	83,	94,	124,	9,	137,	12,	50]

]

# === Step 2. 設定 job 的機台加工順序 ===
mOrder = [
       [4,	2,	1,	5,	3,	0,	6,	11,	10,	12,	8,	7,	13,	9,	14],
       [2,	0,	4,	1,	6,	3,	5,	7,	13,	9,	11,	8,	12,	10,	14],
       [0,	2,	4,	5,	1,	3,	6,	14,	7,	13,	12,	8,	9,	10,	11],
       [4,	2,	6,	5,	3,	0,	1,	14,	12,	9,	11,	10,	8,	13,	7],
       [4,	2,	1,	3,	6,	5,	0,	11,	8,	10,	14,	7,	9,	13,	12],
       [3,	0,	6,	5,	4,	2,	1,	10,	7,	14,	8,	13,	9,	11,	12],
       [0,	1,	5,	3,	2,	6,	4,	8,	7,	14,	11,	12,	9,	10,	13],
       [2,	3,	6,	5,	4,	1,	0,	10,	8,	13,	12,	11,	14,	9,	7],
       [6,	4,	0,	3,	2,	1,	5,	9,	8,	14,	7,	13,	10,	11,	12],
       [0,	1,	4,	2,	5,	3,	6,	10,	7,	13,	11,	9,	8,	12,	14],
       [5,	6,	0,	1,	2,	3,	4,	8,	7,	12,	13,	10,	9,	11,	14],
       [1,	6,	5,	0,	4,	2,	3,	9,	13,	12,	8,	7,	11,	10,	14],
       [6,	3,	2,	0,	5,	1,	4,	7,	14,	11,	8,	12,	13,	10,	9],
       [6,	3,	4,	0,	2,	5,	1,	14,	11,	8,	10,	7,	9,	12,	13],
       [3,	1,	5,	4,	2,	0,	6,	9,	12,	14,	8,	13,	7,	10,	11],
       [5,	3,	1,	0,	2,	4,	6,	10,	12,	9,	8,	11,	14,	7,	13],
       [6,	5,	1,	4,	2,	0,	3,	9,	7,	12,	11,	8,	13,	14,	10],
       [3,	6,	1,	0,	4,	5,	2,	12,	14,	7,	8,	9,	13,	11,	10],
       [4,	6,	3,	1,	5,	0,	2,	13,	9,	11,	8,	14,	10,	7,	12],
       [4,	3,	2,	1,	0,	6,	5,	11,	10,	7,	12,	13,	8,	14,	9]
]


# ==== 參數設定(與演算法相關) ====

NUM_ITERATION = 1000			# 世代數(迴圈數)

NUM_CHROME = 500				# 染色體個數
NUM_BIT = NUM_JOB * NUM_MACHINE		   # 染色體長度 # === Step 3-1. 編碼是 000111222 的排列 ===

Pc = 0.99   					# 交配率 (代表共執行Pc*NUM_CHROME/2次交配)
Pm = 0.01   					# 突變率 (代表共要執行Pm*NUM_CHROME*NUM_BIT次突變)

NUM_PARENT = NUM_CHROME                         # 父母的個數
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)        # 交配的次數
NUM_CROSSOVER_2 = NUM_CROSSOVER*2               # 上數的兩倍
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)   # 突變的次數

#np.random.seed(0)          # 若要每次跑得都不一樣的結果，就把這行註解掉

# ==== 基因演算法會用到的函式 ====    # === Step 5. 設定適應度函數 ===
def initPop():             # 初始化群體
    p = []

    # === 編碼 000111222 的排列  ===
    for i in range(NUM_CHROME) :        
        a = []
        for j in range(NUM_JOB):
            for k in range(NUM_MACHINE):
                a.append(j)
        np.random.shuffle(a)

        p.append(a)
        
    return p

def fitFunc_makespan(x):            # 適應度函數
        
    m_keys=[j for j in range(NUM_MACHINE)]
    j_keys=[j+1 for j in range(NUM_JOB)]
    key_count={key:0 for key in j_keys}
    j_count={key:0 for key in j_keys}
    m_count={key:0 for key in range(NUM_MACHINE)}
    
    for i in x:
        j=i+1
        current_m=int(mOrder[j-1][key_count[j]]) #current_m = 2
        pt=int(pTime[j-1][current_m]) #processing time = 38

        j_count[j]=j_count[j]+pt
        m_count[current_m]=m_count[current_m]+pt

        if m_count[current_m]<j_count[j]:
            m_count[current_m]=j_count[j]
        elif m_count[current_m]>j_count[j]:
            j_count[j]=m_count[current_m]

        key_count[j]=key_count[j]+1
    
    #找max
    maximum=0
    minimum=999999
    
    for m in m_keys:
        if m_count[m]>maximum:
            maximum = m_count[m];
        elif m_count[m]<minimum:
            minimum = m_count[m];
    diff = maximum - minimum
    
    return -maximum           # 因為是最小化問題

def evaluatePop(p):        # 評估群體之適應度
    return [fitFunc_makespan(p[i]) for i in range(len(p))]

def selection(p, p_fit):   # 用二元競爭式選擇法來挑父母
	a = []

	for i in range(NUM_PARENT):
		[j, k] = np.random.choice(NUM_CHROME, 2, replace=False)  # 任選兩個index
		if p_fit[j] > p_fit[k] :                      # 擇優
			a.append(p[j].copy())
		else:
			a.append(p[k].copy())

	return a

def crossover_one_point(p):           # 用單點交配來繁衍子代 (new)
	a = []

	for i in range(NUM_CROSSOVER) :
		c = np.random.randint(1, NUM_BIT)      		  # 隨機找出單點(不包含0)
		[j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index
     
		child1, child2 = p[j].copy(), p[k].copy()
		remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # 存還沒被用掉的城市
       
		for m in range(NUM_BIT):
			if m < c :
				remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
				remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]
		
		t = 0
		for m in range(NUM_BIT):
			if m >= c :
				child1[m] = remain2[t]
				child2[m] = remain1[t]
				t += 1
		
		a.append(child1)
		a.append(child2)

	return a

def mutation(p):	           # 突變
	for _ in range(NUM_MUTATION) :
		row = np.random.randint(NUM_CROSSOVER_2)  # 任選一個染色體
		[j, k] = np.random.choice(NUM_BIT, 2, replace=False)  # 任選兩個基因
      
		p[row][j], p[row][k] = p[row][k], p[row][j]       # 此染色體的兩基因互換


def sortChrome(a, a_fit):	    # a的根據a_fit由大排到小
    a_index = range(len(a))                         # 產生 0, 1, 2, ..., |a|-1 的 list
    
    a_fit, a_index = zip(*sorted(zip(a_fit,a_index), reverse=True)) # a_index 根據 a_fit 的大小由大到小連動的排序
   
    return [a[i] for i in a_index], a_fit           # 根據 a_index 的次序來回傳 a，並把對應的 fit 回傳

def replace(p, p_fit, a, a_fit):            # 適者生存
    b = np.concatenate((p,a), axis=0)               # 把本代 p 和子代 a 合併成 b
    b_fit = p_fit + a_fit                           # 把上述兩代的 fitness 合併成 b_fit
    
    b, b_fit = sortChrome(b, b_fit)                 # b 和 b_fit 連動的排序
    
    return b[:NUM_CHROME], list(b_fit[:NUM_CHROME]) # 回傳 NUM_CHROME 個為新的一個世代


# ==== 主程式 ====

pop = initPop()             # 初始化 pop
pop_fit = evaluatePop(pop)  # 算 pop 的 fit

best_outputs = []                           # 用此變數來紀錄每一個迴圈的最佳解 (new)
best_outputs.append(np.max(pop_fit))        # 存下初始群體的最佳解

mean_outputs = []                           # 用此變數來紀錄每一個迴圈的平均解 (new)
mean_outputs.append(np.average(pop_fit))        # 存下初始群體的最佳解

for i in range(NUM_ITERATION) :
    parent = selection(pop, pop_fit)            # 挑父母
    offspring = crossover_one_point(parent)     # 單點交配
    mutation(offspring)                         # 突變
    offspring_fit = evaluatePop(offspring)      # 算子代的 fit
    pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)    # 取代

    best_outputs.append(np.max(pop_fit))        # 存下這次的最佳解
    mean_outputs.append(np.average(pop_fit))    # 存下這次的平均解

    print('iteration %d: y = %d'	%(i, -pop_fit[0]))     # fit 改負的
    
best_chromosome = pop[0]
print("fitFunc: ", fitFunc_makespan(pop[0]))
print("Best sequence:", best_chromosome)
