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
    [139,	82,	94,	61,	137,	93,	72,	48,	62,	82,	182,	60,	57,	134,	23],
[61,	173,	133,	13,	186,	68,	121,	100,	195,	30,	46,	23,	5,	28,	1],
[164,	162,	195,	126,	149,	123,	179,	16,	185,	53,	23,	160,	145,	68,	41],
[90,	83,	2,	82,	115,	165,	173,	40,	105,	17,	121,	17,	14,	88,	151],
[6,	170,	77,	118,	80,	143,	21,	1,	159,	94,	28,	160,	103,	152,	64],
[41,	115,	35,	76,	159,	26,	9,	140,	35,	41,	23,	109,	105,	181,	16],
[191,	140,	43,	79,	52,	128,	198,	35,	192,	97,	100,	10,	132,	182,	7],
[7,	24,	140,	127,	49,	67,	172,	28,	76,	142,	40,	40,	186,	138,	95],
[96,	65,	25,	158,	101,	168,	126,	200,	182,	45,	125,	119,	104,	128,	122],
[24,	24,	74,	146,	173,	180,	84,	126,	43,	88,	1,	181,	68,	106,	49],
[161,	73,	164,	146,	169,	7,	80,	2,	82,	41,	179,	186,	17,	48,	61],
[194,	147,	117,	7,	98,	147,	93,	112,	186,	138,	96,	11,	154,	17,	190],
[52,	175,	110,	29,	38,	192,	86,	7,	105,	21,	114,	190,	12,	165,	134],
[100,	158,	88,	42,	156,	188,	19,	115,	26,	55,	5,	195,	170,	22,	58],
[77,	183,	161,	36,	31,	199,	186,	119,	132,	62,	192,	192,	129,	68,	148],
[79,	183,	3,	125,	68,	145,	2,	26,	15,	189,	176,	117,	188,	158,	130],
[184,	187,	109,	69,	82,	51,	103,	89,	118,	162,	144,	163,	117,	36,	101],
[88,	146,	194,	18,	161,	49,	55,	120,	193,	168,	97,	115,	152,	57,	188],
[25,	147,	125,	67,	196,	115,	10,	134,	169,	81,	6,	107,	160,	60,	35],
[200,	86,	191,	25,	198,	118,	85,	47,	159,	26,	10,	58,	123,	17,	158]

]

# === Step 2. 設定 job 的機台加工順序 ===
mOrder = [
   [1,	4,	5,	0,	6,	3,	2,	10,	13,	11,	7,	12,	14,	8,	9],
[3,	0,	5,	4,	1,	6,	2,	12,	8,	7,	10,	13,	9,	14,	11],
[6,	0,	5,	1,	2,	4,	3,	9,	10,	13,	8,	11,	7,	12,	14],
[3,	2,	1,	6,	0,	4,	5,	14,	10,	9,	13,	11,	8,	12,	7],
[3,	2,	6,	5,	0,	1,	4,	12,	11,	8,	10,	13,	7,	14,	9],
[6,	4,	1,	3,	5,	0,	2,	8,	11,	12,	7,	9,	10,	14,	13],
[3,	4,	1,	6,	0,	5,	2,	11,	14,	13,	12,	10,	9,	8,	7],
[1,	6,	5,	2,	4,	3,	0,	11,	10,	7,	14,	13,	12,	8,	9],
[6,	3,	0,	2,	1,	4,	5,	12,	7,	9,	11,	10,	14,	13,	8],
[1,	6,	2,	3,	5,	0,	4,	10,	7,	11,	8,	13,	9,	12,	14],
[1,	3,	4,	2,	0,	6,	5,	10,	11,	13,	12,	9,	14,	8,	7],
[0,	3,	2,	4,	5,	6,	1,	9,	13,	11,	7,	10,	8,	14,	12],
[5,	2,	6,	4,	1,	0,	3,	13,	14,	8,	11,	7,	9,	10,	12],
[2,	5,	0,	1,	4,	3,	6,	14,	11,	12,	13,	8,	7,	9,	10],
[6,	0,	3,	2,	5,	4,	1,	11,	14,	9,	8,	7,	13,	10,	12],
[1,	6,	2,	3,	0,	4,	5,	11,	14,	8,	7,	9,	10,	13,	12],
[1,	3,	4,	5,	6,	0,	2,	10,	11,	12,	7,	13,	14,	8,	9],
[3,	0,	4,	5,	1,	6,	2,	11,	14,	13,	7,	8,	10,	9,	12],
[4,	6,	5,	2,	1,	0,	3,	8,	11,	10,	12,	7,	14,	9,	13],
[3,	2,	6,	5,	0,	4,	1,	12,	14,	8,	10,	13,	7,	9,	11]

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
