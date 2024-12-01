# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:20:28 2024

@author: user
"""

# machine 8~machine 15 (7~14) 共8台

import numpy as np
#import math

# ==== 參數設定(與問題相關) ====

NUM_JOB = 20            # 工件個數   # === Step 1. 設定成 20 個工件 ===
END_JOBNUM = 8  #7~14共8台
NUM_MACHINE = 8                     # === Step 1. 設定 15 個機台

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

NUM_ITERATION =100	# 世代數(迴圈數)

NUM_CHROME = 1				# 染色體個數
NUM_BIT = NUM_JOB * NUM_MACHINE		   # 染色體長度 # === Step 3-1. 編碼是 000111222 的排列 ===


# === Step 3-2. NUM_BIT 要修改成 3 x 3 ===

#np.random.seed(0)          # 若要每次跑得都不一樣的結果，就把這行註解掉

# ==== 基因演算法會用到的函式 ====    # === Step 5. 設定適應度函數 ===
def initPop():             # 初始化群體
    p = []
    for j in range(END_JOBNUM): #0~7
        for k in range(NUM_MACHINE):
            p.append(7+j)  #7~14
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
    [j, k] = np.random.choice(END_JOBNUM * NUM_MACHINE, 2, replace=False)  # 任選兩個基因
    a[j], a[k] = a[k], a[j]       # 此染色體的兩基因互換
    return a

# ==== 主程式 ====
sequence = []  #前面part排出來的解
pop = initPop()             # 初始化 pop
print("End Part: ",pop)
best_pop = pop
best_fit = fitFunc(sequence + pop)

for i in range(NUM_ITERATION) :
    
    after_pop = mutation_together(pop) # 算 pop 的 fit
    after_fit = fitFunc(sequence + after_pop)  # 算 pop 的 fit
   # print("before: ", best_fit)
    #print("after: ", after_fit)
    if best_fit < after_fit: #換完比較好
        best_pop = after_pop
        best_fit = after_fit
    print("iteration", i,": ", (-1)*best_fit)
    
print("Best sequence: ", best_pop)

