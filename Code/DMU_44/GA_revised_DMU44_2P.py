# -*- coding: utf-8 -*-
"""
#GA_revised_two_point
#DMU 44
#適應度值為 makespan

Created on Wed Jun 19 12:28:10 2024

@author: user
"""
import numpy as np
# import math


# ==== 參數設定(與問題相關) ====

NUM_JOB = 20            # 工件個數   # === Step 1. 設定成 20 個工件 ===
NUM_MACHINE = 15                     # === Step 1. 設定 15 個機台

# === Step 2. 設定 processing time 矩陣(row: job ID; col: 對應其加工機台所用的加工時間) ===
pTime = [
     [142,	135,	12,	170,	134,	198,	21,	21,	121,	33,	130,	47,	166,	96,	42],
[146,	5,	153,	129,	39,	191,	148,	173,	152,	172,	113,	125,	161,	147,	74],
[111,	185,	64,	93,	51,	137,	8,	90,	182,	110,	165,	18,	160,	192,	6],
[191,	191,	26,	119,	121,	197,	153,	179,	88,	77,	131,	44,	185,	38,	163],
[47,	164,	105,	15,	125,	130,	132,	194,	70,	48,	31,	159,	82,	172,	97],
[78,	62,	81,	116,	182,	10,	176,	135,	13,	142,	189,	24,	98,	157,	66],
[44,	180,	129,	28,	181,	190,	33,	22,	193,	93,	51,	60,	74,	88,	103],
[152,	143,	34,	63,	47,	177,	72,	148,	178,	149,	146,	166,	2,	183,	10],
[53,	150,	139,	34,	114,	42,	157,	76,	147,	27,	177,	23,	31,	171,	109],
[50,	50,	192,	63,	166,	179,	39,	111,	136,	12,	35,	57,	3,	55,	142],
[41,	68,	165,	183,	53,	74,	56,	1,	80,	77,	97,	112,	33,	31,	168],
[200,	157,	41,	170,	100,	2,	113,	147,	62,	168,	48,	121,	77,	76,	36],
[56,	15,	133,	85,	180,	89,	98,	136,	194,	173,	166,	167,	171,	160,	152],
[49,	35,	160,	32,	22,	174,	64,	173,	39,	2,	5,	50,	137,	190,	128],
[187,	19,	115,	76,	151,	106,	102,	184,	94,	68,	49,	179,	137,	54,	163],
[127,	26,	183,	10,	64,	5,	73,	86,	45,	200,	2,	44,	46,	97,	184],
[169,	84,	87,	179,	126,	142,	12,	133,	39,	168,	121,	179,	24,	142,	147],
[77,	113,	97,	17,	139,	41,	98,	170,	48,	116,	172,	60,	165,	196,	77],
[191,	3,	5,	189,	35,	195,	55,	199,	100,	33,	39,	131,	69,	169,	158],
[78,	175,	50,	135,	146,	24,	131,	152,	94,	99,	65,	100,	124,	70,	152]
]

# === Step 2. 設定 job 的機台加工順序 ===
mOrder = [
   [0,	5,	1,	3,	2,	6,	4,	7,	9,	11,	10,	14,	13,	8,	12],
[2,	4,	3,	6,	1,	0,	5,	12,	14,	7,	10,	13,	9,	8,	11],
[2,	5,	3,	6,	1,	4,	0,	13,	10,	8,	7,	14,	9,	12,	11],
[6,	5,	0,	3,	1,	2,	4,	10,	7,	11,	14,	8,	12,	13,	9],
[2,	3,	4,	6,	5,	0,	1,	8,	9,	11,	14,	13,	7,	10,	12],
[1,	3,	4,	2,	6,	0,	5,	11,	13,	14,	8,	9,	10,	7,	12],
[3,	0,	5,	1,	2,	4,	6,	10,	11,	8,	12,	13,	14,	7,	9],
[5,	4,	6,	2,	0,	1,	3,	10,	13,	12,	11,	7,	14,	8,	9],
[6,	2,	0,	3,	5,	1,	4,	11,	13,	10,	8,	9,	14,	12,	7],
[3,	2,	4,	6,	0,	1,	5,	7,	10,	14,	8,	11,	12,	9,	13],
[3,	1,	0,	6,	2,	5,	4,	13,	8,	9,	14,	12,	10,	7,	11],
[1,	0,	6,	4,	5,	3,	2,	7,	10,	8,	9,	14,	12,	13,	11],
[2,	3,	5,	6,	0,	4,	1,	10,	13,	9,	8,	14,	7,	11,	12],
[0,	6,	5,	4,	3,	2,	1,	13,	12,	7,	9,	14,	10,	11,	8],
[2,	5,	1,	4,	3,	6,	0,	11,	7,	10,	9,	14,	12,	8,	13],
[0,	2,	1,	3,	6,	4,	5,	11,	10,	8,	12,	14,	13,	9,	7],
[1,	2,	3,	5,	6,	4,	0,	8,	9,	12,	13,	7,	14,	10,	11],
[6,	0,	5,	1,	2,	4,	3,	7,	9,	11,	12,	10,	14,	8,	13],
[4,	5,	1,	0,	6,	3,	2,	9,	11,	10,	14,	12,	8,	7,	13],
[6,	4,	1,	5,	2,	3,	0,	13,	7,	9,	14,	10,	12,	8,	11]

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
    offspring = crossover_two_point(parent)     # 雙點交配
    mutation(offspring)                         # 突變
    offspring_fit = evaluatePop(offspring)      # 算子代的 fit
    pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)    # 取代

    best_outputs.append(np.max(pop_fit))        # 存下這次的最佳解
    mean_outputs.append(np.average(pop_fit))    # 存下這次的平均解

    print('iteration %d: y = %d'	%(i, -pop_fit[0]))     # fit 改負的
    
best_chromosome = pop[0]
print("fitFunc: ", fitFunc_makespan(pop[0]))
print("Best sequence:", best_chromosome)
