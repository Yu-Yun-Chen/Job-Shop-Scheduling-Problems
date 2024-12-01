# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 14:24:35 2024

@author: user
"""
# machine 1~machine 7 (0~6) 共7台
import numpy as np
#import math

# ==== 參數設定(與問題相關) ====

NUM_JOB = 20            # 工件個數   # === Step 1. 設定成 20 個工件 ===
NUM_MACHINE = 7                     # === Step 1. 設定 15 個機台

# === Step 2. 設定 processing time 矩陣(row: job ID; col: 對應其加工機台所用的加工時間) ===
pTime = [
           [151,190,38,25,109,63,188],
           [34,171,163,96,72,48,177],
           [33,9,66,143,135,12,47],
           [108,147,160,161,145,167,84],
           [44,183,79,96,52,192,125],
           [191,8,31,30,120,120,39],
           [175,172,89,58,160,166,54],
           [195,28,84,193,105,87,15],
           [150,102,136,99,43,115,44],
           [91,152,11,173,21,159,50],
           [185,140,189,59,79,123,111],
           [95,108,50,183,14,184,142],
           [186,22,143,64,85,146,178],
           [16,114,121,169,150,15,147],
           [81,18,108,124,55,105,137],
           [34,193,51,149,61,62,132],
           [135,75,104,67,32,74,61],
           [56,79,30,6,74,68,78],
           [180,30,83,74,13,57,89],
           [21,30,132,35,72,98,104]

]

# === Step 2. 設定 job 的機台加工順序 ===
mOrder = [
    [2,5,4,0,1,3,6],
    [5,6,4,0,3,1,2],
    [2,4,1,6,0,5,3],
    [3,5,2,0,1,6,4],
    [6,4,0,5,2,1,3],
    [2,0,3,5,1,6,4],
    [2,6,5,4,1,3,0],
    [3,0,1,2,4,5,6],
    [2,1,4,0,3,6,5],
    [3,5,2,1,6,0,4],
    [0,1,3,2,4,5,6],
    [5,6,2,1,0,3,4],
    [5,2,3,4,0,1,6],
    [4,2,1,0,3,6,5],
    [4,3,2,6,5,1,0],
    [5,1,2,6,0,3,4],
    [3,1,2,4,0,5,6],
    [5,2,3,1,4,6,0],
    [2,4,5,0,1,6,3],
    [4,2,0,5,6,1,3]

]


# ==== 參數設定(與演算法相關) ====

NUM_ITERATION =1000000		# 世代數(迴圈數)

NUM_CHROME = 1				# 染色體個數
NUM_BIT = NUM_JOB * NUM_MACHINE		   # 染色體長度 # === Step 3-1. 編碼是 000111222 的排列 ===
#CUT = 6
#FRONT = NUM_JOB*(CUT+1)  #前段的基因數 20*7
#BACK = NUM_JOB*(CUT+2)   #後段的基因數 20*8

Pc = 0.99   					# 交配率 (代表共執行Pc*NUM_CHROME/2次交配)
Pm = 0.01   					# 突變率 (代表共要執行Pm*NUM_CHROME*NUM_BIT次突變)

NUM_PARENT = NUM_CHROME                         # 父母的個數
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)        # 交配的次數
NUM_CROSSOVER_2 = NUM_CROSSOVER*2               # 上數的兩倍
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)   # 突變的次數
# === Step 3-2. NUM_BIT 要修改成 3 x 3 ===

#np.random.seed(0)          # 若要每次跑得都不一樣的結果，就把這行註解掉

# ==== 基因演算法會用到的函式 ====    # === Step 5. 設定適應度函數 ===
def initPop():             # 初始化群體
    p = []
    for j in range(NUM_JOB): #0~6
        for k in range(NUM_MACHINE):
            p.append(j)
        np.random.shuffle(p)

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


'''
def mutation_separate(p):	           # 分開交換
    rannum = np.random.randint (0,2) #產出0/1
    if rannum == 0: #交換前段就好
        [j, k] = np.random.choice(FRONT, 2, replace=False)
        p[j], p[k] = p[k], p[j]       # 此染色體的兩基因互換

    if rannum == 1: #交換後段就好
        [j, k] = np.random.choice(BACK, 2, replace=False)
        p[FRONT+j], p[FRONT+k] = p[FRONT+k], p[FRONT+j]       # 此染色體的兩基因互換
'''

def mutation_together(p):  #整體交換
    a = []
    a = p
    [j, k] = np.random.choice(NUM_BIT, 2, replace=False)  # 任選兩個基因
    a[j], a[k] = a[k], a[j]       # 此染色體的兩基因互換
    return a

# ==== 主程式 ====

pop = initPop()             # 初始化 pop
print("First population: ",pop)
best_pop = pop
best_fit = fitFunc(pop)

for i in range(NUM_ITERATION) :
    
    after_pop = mutation_together(pop) # 算 pop 的 fit
    after_fit = fitFunc(after_pop)  # 算 pop 的 fit
   # print("before: ", best_fit)
    #print("after: ", after_fit)
    if best_fit < after_fit: #換完比較好
        best_pop = after_pop
        best_fit = after_fit
    print("iteration", i,": ", (-1)*best_fit)
    
print("Best sequence: ", best_pop)

