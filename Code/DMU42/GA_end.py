# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:49:01 2024

@author: user
#GA_END
"""
# machine 8~machine 15 (7~14) 共8台
import numpy as np
#import math

# ==== 參數設定(與問題相關) ====

NUM_JOB = 20            # 工件個數   # === Step 1. 設定成 20 個工件 ===
END_MACHINE = 8  #7~14共8台
NUM_MACHINE = 15                     # === Step 1. 設定 15 個機台

# === Step 2. 設定 processing time 矩陣(row: job ID; col: 對應其加工機台所用的加工時間) ===
pTime = [
           [151,190,38,25,109,63,188,61,80,161,10,127,128,66,170],
           [34,171,163,96,72,48,177,123,200,81,101,193,70,183,98],
           [33,9,66,143,135,12,47,146,41,71,17,67,91,13,135],
           [108,147,160,161,145,167,84,169,74,91,144,93,96,90,25],
           [44,183,79,96,52,192,125,50,66,160,88,166,21,85,41],
           [191,8,31,30,120,120,39,149,30,83,19,136,180,136,197],
           [175,172,89,58,160,166,54,89,62,5,175,180,152,45,184],
           [195,28,84,193,105,87,15,180,192,170,106,1,179,37,195],
           [150,102,136,99,43,115,44,29,152,18,95,169,5,164,46],
           [91,152,11,173,21,159,50,115,171,158,174,107,184,80,128],
           [185,140,189,59,79,123,111,122,155,30,44,151,89,188,18],
           [95,108,50,183,14,184,142,88,197,52,193,145,84,113,113],
           [186,22,143,64,85,146,178,104,163,115,83,57,57,61,181],
           [16,114,121,169,150,15,147,167,175,59,94,159,76,153,76],
           [81,18,108,124,55,105,137,63,136,169,200,138,69,9,171],
           [34,193,51,149,61,62,132,77,31,80,54,166,50,198,70],
           [135,75,104,67,32,74,61,198,165,38,15,39,1,136,115],
           [56,79,30,6,74,68,78,178,66,137,97,99,198,175,180],
           [180,30,83,74,13,57,89,96,199,78,169,121,156,27,54],
           [21,30,132,35,72,98,104,195,149,40,198,110,131,35,15]

]

# === Step 2. 設定 job 的機台加工順序 ===
mOrder = [
    [2,5,4,0,1,3,6,13,7,11,9,10,14,8,12],
    [5,6,4,0,3,1,2,7,8,9,12,13,14,10,11],
    [2,4,1,6,0,5,3,10,12,14,11,9,7,8,13],
    [3,5,2,0,1,6,4,9,14,13,8,11,7,12,10],
    [6,4,0,5,2,1,3,7,8,14,9,10,13,12,11],
    [2,0,3,5,1,6,4,7,11,13,12,8,10,14,9],
    [2,6,5,4,1,3,0,11,10,13,14,9,8,7,12],
    [3,0,1,2,4,5,6,11,9,10,8,13,12,14,7],
    [2,1,4,0,3,6,5,7,8,9,10,11,12,13,14],
    [3,5,2,1,6,0,4,11,7,14,12,8,10,13,9],
    [0,1,3,2,4,5,6,11,10,7,12,14,9,13,8],
    [5,6,2,1,0,3,4,13,7,14,8,10,9,12,11],
    [5,2,3,4,0,1,6,11,14,9,12,10,13,7,8],
    [4,2,1,0,3,6,5,11,9,10,14,12,8,13,7],
    [4,3,2,6,5,1,0,8,10,9,11,14,13,12,7],
    [5,1,2,6,0,3,4,12,8,9,14,10,11,13,7],
    [3,1,2,4,0,5,6,14,11,10,13,7,12,8,9],
    [5,2,3,1,4,6,0,8,11,9,7,12,13,10,14],
    [2,4,5,0,1,6,3,9,10,11,8,7,13,14,12],
    [4,2,0,5,6,1,3,7,9,10,8,14,13,12,11]

]


# ==== 參數設定(與演算法相關) ====

NUM_ITERATION = 900			# 世代數(迴圈數)

NUM_CHROME = 1000				# 染色體個數
NUM_BIT = NUM_JOB * NUM_MACHINE		   # 染色體長度 # === Step 3-1. 編碼是 000111222 的排列 ===
NUM_END = NUM_JOB * END_MACHINE
Pc = 0.99  					# 交配率 (代表共執行Pc*NUM_CHROME/2次交配)
Pm = 0.005   					# 突變率 (代表共要執行Pm*NUM_CHROME*NUM_BIT次突變)

NUM_PARENT = NUM_CHROME                         # 父母的個數
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)        # 交配的次數
NUM_CROSSOVER_2 = NUM_CROSSOVER*2               # 上數的兩倍
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_END)   # 突變的次數
# === Step 3-2. NUM_BIT 要修改成 3 x 3 ===

#np.random.seed(0)          # 若要每次跑得都不一樣的結果，就把這行註解掉

# ==== 基因演算法會用到的函式 ====    # === Step 5. 設定適應度函數 ===
def initPop():             # 初始化群體
    p = []

    # === 編碼 000111222 的排列  ===
    for i in range(NUM_CHROME) :        
        a = []
        for j in range(NUM_JOB):
            for k in range(END_MACHINE):
                a.append(j)
        np.random.shuffle(a)

        p.append(a)
        
    return p

def fitFunc(a):            # 適應度函數
    
    sequence=[13,10,15,15,2,5,14,11,2,7,4,17,11,5,8,19,9,1,16,7,15,11,1,5,19,12,5,19,2,10,10,9,12,0,8,8,15,5,7,2,3,8,15,5,0,16,1,9,8,11,12,17,6,19,4,0,19,15,10,9,1,2,11,6,18,13,3,0,12,6,4,15,0,18,14,6,10,11,3,17,0,11,1,8,19,14,9,18,18,6,18,16,3,13,4,12,16,6,10,0,3,19,17,1,13,10,17,7,14,17,14,16,14,17,12,7,4,1,2,9,5,6,18,13,2,13,7,3,9,13,8,12,16,16,14,18,7,4,4,3]
    x = np.concatenate((sequence, a), axis=None)
   
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
		c = np.random.randint(1, NUM_END)      		  # 隨機找出單點(不包含0)
		[j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index
     
		child1, child2 = p[j].copy(), p[k].copy()
		remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # 存還沒被用掉的城市
       
		for m in range(NUM_END):
			if m < c :
				remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
				remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]
		
		t = 0
		for m in range(NUM_END):
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
		mask = np.random.randint(2, size=NUM_END)
		[j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index
     
		child1, child2 = p[j].copy(), p[k].copy()
		remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # 存還沒被用掉的城市
       
		for m in range(NUM_END):
			if mask[m] == 1 :
				remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
				remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]
		
		t = 0
		for m in range(NUM_END):
			if mask[m] == 0 :
				child1[m] = remain2[t]
				child2[m] = remain1[t]
				t += 1
		
		a.append(child1)
		a.append(child2)

	return a
def mutation(p):	           # 突變
	for _ in range(NUM_MUTATION) :
		row = np.random.randint(NUM_CROSSOVER_2)  # 任選一個染色體
		[j, k] = np.random.choice(NUM_END, 2, replace=False)  # 任選兩個基因
      
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

sequence=[13,10,15,15,2,5,14,11,2,7,4,17,11,5,8,19,9,1,16,7,15,11,1,5,19,12,5,19,2,10,10,9,12,0,8,8,15,5,7,2,3,8,15,5,0,16,1,9,8,11,12,17,6,19,4,0,19,15,10,9,1,2,11,6,18,13,3,0,12,6,4,15,0,18,14,6,10,11,3,17,0,11,1,8,19,14,9,18,18,6,18,16,3,13,4,12,16,6,10,0,3,19,17,1,13,10,17,7,14,17,14,16,14,17,12,7,4,1,2,9,5,6,18,13,2,13,7,3,9,13,8,12,16,16,14,18,7,4,4,3]


for i in range(NUM_ITERATION) :
    parent = selection(pop, pop_fit)            # 挑父母
    offspring = crossover_one_point(parent)     # 單點交配
    mutation(offspring)                         # 突變
    offspring_fit = evaluatePop(offspring)      # 算子代的 fit
    pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)    # 取代

    best_outputs.append(np.max(pop_fit))        # 存下這次的最佳解
    mean_outputs.append(np.average(pop_fit))    # 存下這次的平均解

    print('iteration %d: y = %d'	%(i, -pop_fit[0]))     # fit 改負的
    
#print("End sequence: ", pop[0])
best_chromosome = np.concatenate((sequence, pop[0]))
print("Best sequence:", best_chromosome)
