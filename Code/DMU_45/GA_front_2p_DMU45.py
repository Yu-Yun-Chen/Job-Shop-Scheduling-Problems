# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:53:34 2024

@author: user
"""
#GA_front
# machine 1~machine 7 (0~6) 共7台
import numpy as np
#import math

# ==== 參數設定(與問題相關) ====

NUM_JOB = 20            # 工件個數   # === Step 1. 設定成 20 個工件 ===
NUM_MACHINE = 7                     # === Step 1. 設定 15 個機台

# === Step 2. 設定 processing time 矩陣(row: job ID; col: 對應其加工機台所用的加工時間) ===
pTime = [
 [48,	54,	90,	114,	159,	16,	105],
[45,	82,	151,	40,	103,	24,	29],
[26,	135,	190,	120,	40,	4,	126],
[140,	21,	194,	173,	74,	9,	169],
[22,	160,	196,	164,	33,	125,	6],
[118,	33,	35,	89,	75,	117,	174],
[11,	79,	15,	141,	134,	78,	141],
[101,	9,	98,	135,	74,	172,	2],
[161,	110,	135,	33,	196,	95,	156],
[166,	151,	30,	59,	122,	99,	165],
[57,	154,	67,	11,	21,	24,	125],
[88,	54,	182,	123,	94,	118,	74],
[108,	30,	15,	163,	107,	118,	126],
[148,	127,	151,	116,	1,	158,	131],
[94,	98,	5,	31,	50,	174,	125],
[173,	198,	164,	169,	46,	151,	65],
[174,	115,	165,	180,	100,	122,	185],
[187,	13,	42,	95,	103,	188,	15],
[134,	40,	42,	132,	1,	177,	13],
[3,	79,	110,	117,	76,	111,	167]
]

# === Step 2. 設定 job 的機台加工順序 ===
mOrder = [
   [6,	5,	0,	3,	1,	2,	4],
[6,	1,	3,	0,	2,	4,	5],
[0,	2,	4,	1,	3,	5,	6],
[0,	1,	4,	2,	5,	3,	6],
[4,	2,	1,	6,	0,	3,	5],
[3,	6,	4,	2,	0,	5,	1],
[6,	3,	0,	2,	1,	5,	4],
[1,	2,	5,	0,	6,	4,	3],
[6,	2,	5,	0,	1,	4,	3],
[2,	0,	5,	6,	3,	1,	4],
[2,	1,	6,	0,	5,	4,	3],
[5,	1,	6,	4,	3,	0,	2],
[2,	0,	3,	5,	4,	1,	6],
[1,	2,	4,	3,	5,	0,	6],
[6,	2,	3,	4,	5,	0,	1],
[1,	6,	0,	3,	4,	2,	5],
[0,	2,	3,	1,	6,	4,	5],
[3,	6,	1,	4,	2,	0,	5],
[6,	0,	5,	1,	3,	4,	2],
[5,	6,	3,	0,	4,	2,	1]

]

# ==== 參數設定(與演算法相關) ====

NUM_ITERATION = 500			# 世代數(迴圈數)

NUM_CHROME = 2000				# 染色體個數
NUM_BIT = NUM_JOB * NUM_MACHINE		   # 染色體長度 # === Step 3-1. 編碼是 000111222 的排列 ===

Pc = 0.99  					# 交配率 (代表共執行Pc*NUM_CHROME/2次交配)
Pm = 0.01   					# 突變率 (代表共要執行Pm*NUM_CHROME*NUM_BIT次突變)

NUM_PARENT = NUM_CHROME                         # 父母的個數
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)        # 交配的次數
NUM_CROSSOVER_2 = NUM_CROSSOVER*2               # 上數的兩倍
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)   # 突變的次數
# === Step 3-2. NUM_BIT 要修改成 3 x 3 ===

np.random.seed(0)          # 若要每次跑得都不一樣的結果，就把這行註解掉

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

def fitFunc(x):            # 適應度函數
    S = np.zeros((NUM_JOB, NUM_MACHINE))    # S[i][j] = Starting time of job i at machine j
    C = np.zeros((NUM_JOB, NUM_MACHINE))    # C[i][j] = Completion time of job i at machine j
    
    B = np.zeros(NUM_MACHINE, dtype=int)    # B[j] = Available time of machine j
    
    opJob = np.zeros(NUM_JOB, dtype=int)    # opJob[i] = current operation ID of job i
    
    for i in range(NUM_BIT):
        m = mOrder[x[i]][opJob[x[i]]]
        if opJob[x[i]] != 0:
            S[x[i]][m] = max([B[m], C[x[i]][mOrder[x[i]][opJob[x[i]]-1]]])
        else:
            S[x[i]][m] = B[m]
            
        C[x[i]][m] = B[m] = S[x[i]][m] + pTime[x[i]][opJob[x[i]]]
        
        opJob[x[i]] += 1
            
    return -max(B)           # 因為是最小化問題

def evaluatePop(p):        # 評估群體之適應度
    return [fitFunc(p[i]) for i in range(len(p))]

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

def crossover_uniform(p):           # 用均勻交配來繁衍子代 (new)
	a = []

	for i in range(NUM_CROSSOVER) :
		mask = np.random.randint(2, size=NUM_BIT)
		[j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index
     
		child1, child2 = p[j].copy(), p[k].copy()
		remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # 存還沒被用掉的城市
       
		for m in range(NUM_BIT):
			if mask[m] == 1 :
				remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
				remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]
		
		t = 0
		for m in range(NUM_BIT):
			if mask[m] == 0 :
				child1[m] = remain2[t]
				child2[m] = remain1[t]
				t += 1
		
		a.append(child1)
		a.append(child2)

	return a

def crossover_two_point(p):           # 用雙點交配來繁衍子代 (new)
	a = []

	for i in range(NUM_CROSSOVER) :
		points = np.random.choice(range(1, NUM_BIT), 2, replace=False) # 隨機找出兩個交配點(不包含0)
		points.sort() # 確保點的順序
		c1, c2 = points

		[j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index

		child1, child2 = p[j].copy(), p[k].copy()
		remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # 存還沒被用掉的城市

		for m in range(NUM_BIT):
			if m < c1 or m >= c2:
				remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
				remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]

		t = 0
		for m in range(NUM_BIT):
			if m >= c1 and m < c2:
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
    offspring = crossover_two_point(parent)     # 單點交配
    mutation(offspring)                         # 突變
    offspring_fit = evaluatePop(offspring)      # 算子代的 fit
    pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)    # 取代

    best_outputs.append(np.max(pop_fit))        # 存下這次的最佳解
    mean_outputs.append(np.average(pop_fit))    # 存下這次的平均解

    print('iteration %d: y = %d'	%(i, -pop_fit[0]))     # fit 改負的
    
best_chromosome = pop[0]
print("Best sequence:", best_chromosome)

