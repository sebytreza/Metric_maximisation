import numpy as np
import matplotlib.pyplot as plt
import pandas as p
import numba
from numba import cuda
from tqdm import tqdm
import math
from time import perf_counter

## SCORE CALCULATION BASED ON SET VALUE METRICS ##
@numba.jit(fastmath=True, nogil=True, parallel = True)
def score(pred,target, func, kwargs = {} ):
    VP = np.sum(np.logical_and(pred,target))
    FP = np.sum(pred) - VP
    FN = np.sum(target) - VP
    TN = len(pred) - VP - FP - FN
    return func(VP, FP, TN, FN, **kwargs)

## DEFINE OF UTILITY FUNCTIONS ##
@numba.njit(fastmath=True, nogil=True)
def f_b(TP, FP, TN, FN, **kwargs):
    b = kwargs.get('b', 1)
    return (1+b**2)*TP/((1+b**2)*TP + b*FN + FP)

@numba.njit(fastmath=True, nogil=True)
def jaccard(TP, FP, TN, FN, **kwargs):
    return TP/(TP + FP + FN)

## MAXIMIZATION OF THE SCORE ##
@numba.njit(fastmath=True, nogil=True, parallel = True)
def max_func(p, **kwargs):
    func = kwargs.get('func', f_b)
    kwargs = kwargs.get('kwargs')

    n = len(p)
    Umax, kmax, U  = 0, 0, 0
    C = np.zeros((n+1,n+1))
    C[0,0] = 1
    for i in range(n):
        C[i+1,0] = (1 - p[i])*C[i,0]
        for j in range(i+1):
            C[i+1,j+1] = p[i]*C[i,j] + (1 - p[i])*C[i,j+1]
              
    S = np.zeros((n+1,n+1))
    S[0,0] = 1
    for i in range(n): 
        S[i+1,0] = (1 - p[n-i-1])*S[i,0]
        for j in range(i+1):
            S[i+1,j+1] = p[n-i-1]*S[i,j] + (1 - p[n-i-1])*S[i,j+1]     

    K = 1
    while K < n and Umax == U:
        U = 0
        for i in range(K+1):
            for j in range(n - K + 1):
                U += C[K,i]*S[n-K,j]*func(i, K -i, n-K-j, j, **kwargs) #func(tp,fp,tn, fn)
            
        if U >= Umax :
            Umax, kmax  = U, K
        K += 1
    return kmax, Umax #return the number of species to keep and the score

## BASELINE STRATEGIES ##
@numba.njit(fastmath=True, nogil=True)
def threshold(p, **kwargs) :
    th  = kwargs.get('th', 0.5)
    K = 0
    while p[K] >= th and K < len(p):
        K += 1
    return K+1

@numba.njit(fastmath=True, nogil=True)
def sum_th(p, **kwargs) :
    return int(np.sum(p)) + 1

@numba.njit(fastmath=True, nogil = True, parallel=True)
def iterate(SOL, PROBAS, P_CALIB, T_CALIB):
    S = len(SOL)
    N = len(SOL[0])
    ST = len(T_CALIB)

    func = f_b
    kwargs = {'b': 1}

    SCORE = np.zeros((S, 7))
    KLIST = np.zeros((S, 7))

    OUTPUT = None

    # Calibrate strategies on the given data #
    
    # TopK
    Ktopk = 0
    Ut_topk = 0.
    Ut_topk2 = 0.
    topk = np.zeros(N)
    while Ut_topk2 >= Ut_topk:
        Ut_topk = Ut_topk2
        Ut_topk2 = 0.
        for i in range(ST):
            probas = P_CALIB[i]
            tar = T_CALIB[i]
            sort = np.argsort(-probas)
            mask = np.where(probas > 0)[0]
            input = probas[mask][sort]
            topk[sort[Ktopk]] = 1
            Ut_topk2 += score(topk,tar,func, **kwargs)
        Ktopk += 1

    # Global threshold
    th = 1.
    e = 0.1
    emax = 0.01
    while e >= emax:
        Ut_th = 0.
        Ut_th2 = 0.
        while Ut_th2 >= Ut_th :
            th = th - e
            Ut_th = Ut_th2
            Ut_th2 = 0.
            for i in range(ST):
                probas = P_CALIB[i]
                tar = T_CALIB[i]
                sort = np.argsort(-probas)
                mask = np.where(probas > 0)[0]
                input = probas[mask][sort]
                Kth = threshold(input, th = th)
                nth = np.zeros(N)
                nth[sort[:Kth]] = 1
                Ut_th2 += score(nth,tar,func, **kwargs)
        th = th + e
        e /= 10

    # Low threshold
    thlow = np.ones(N)
    for i in range(N):
        for j in range(ST):
            if T_CALIB[j,i] == 1:
                thlow[i] = min(thlow[i], P_CALIB[j,i])

    # 5% threshold
    th5pc = np.ones(N)
    for i in range(N):
        th5pc[i] = np.percentile(P_CALIB[np.where(T_CALIB[:,i]),i], 5)

    del P_CALIB
    del T_CALIB 

    # Iterate throught the dataset #
    for i in range(S):

        probas = PROBAS[i]
        tar = SOL[i]
        
        sort = np.argsort(-probas)
        mask = np.where(input > 0)[0] #clip species with null probability
        input = probas[mask][sort]

        K, _ = max_func(input, func, kwargs)
        pred = np.zeros(N).astype(np.intp)
        pred[sort[:K]] = 1
        U = score(pred,tar,func, **kwargs)

        
        Ktopk = 8
        topk = np.zeros(N)
        topk[sort[:Ktopk]] = 1
        U_topk = score(topk,tar,func, **kwargs)
    
        K0_5 = threshold(input, th = 0.5)
        n0_5 = np.zeros(N)
        n0_5[sort[:K0_5]] = 1
        U_0_5 = score(n0_5,tar,func, **kwargs)

        th  = 0.5
        Kth = threshold(input, th)
        nth = np.zeros(N)
        nth[sort[:Kth]] = 1
        U_th = score(nth,tar,func, **kwargs)

        thlow = 0.1
        Klow = threshold(input, thlow)
        nlow = np.zeros(N)
        nlow[sort[:Klow]] = 1
        U_low = score(nlow,tar,func, **kwargs)

        th5pc = 0.1
        K5pc = threshold(input, th5pc)
        n5pc = np.zeros(N)
        n5pc[sort[:K5pc]] = 1
        U_5pc = score(n5pc,tar,func, **kwargs)

        Ksum = sum_th(input)
        nsum = np.zeros(N)
        nsum[sort[:Ksum]] = 1
        U_sum = score(nsum,tar,func, **kwargs)

        KLIST[i] = np.array([K, Ktopk, K0_5, Kth, Klow, K5pc, Ksum])
        SCORE[i] = np.array([U, U_topk, U_0_5, U_th, U_low, U_5pc, U_sum])

        output = np.where(pred == 1)

        if OUTPUT is None:
            OUTPUT = output
        else:
            OUTPUT = np.vstack((OUTPUT, output))

    print("Maximization :" + np.mean(U[:,0]))
    print("Topk         :" + np.mean(U_topk[:,1]))
    print("Th_0.5       :" + np.mean(U_0_5[:,2]))
    print("Th_opti      :" + np.mean(U_th[:,3]))
    print("Th_lowest    :" + np.mean(U_low[:,4]))
    print("Th_5%        :" + np.mean(U_5pc[:,5]))
    print("Sum          :" + np.mean(U_sum[:,6]))

    return KLIST, OUTPUT
 

def main(pval = 0.2):
    sol_file = 'data/hmsc_test_species.csv'
    pred_file = 'data/hmsc_test_probas.csv'
    tcalib_file = 'sol_file_test.csv'
    pcalib_file = 'pred_file_test.csv'

    sol = p.read_csv(sol_file)
    probas = p.read_csv(pred_file, delimiter = ',')
    probas = probas.join(sol.set_index('surveyId'), on='surveyId')
    surveys = probas['surveyId']
    probas = probas.dropna()
    sol = probas['speciesId'].to_numpy(dtype=str)
    PROBAS = probas.drop(columns = ['surveyId', 'speciesId']).to_numpy(dtype=np.float32)
    del probas
    if pval == 0:
        tcalib  = p.read_csv(tcalib_file)
        pcalib  = p.read_csv(pcalib_file, delimiter = ',')
        pcalib = pcalib.join(tcalib.set_index('surveyId'), on='surveyId')
        pcalib = pcalib.dropna()
        tcalib = pcalib['speciesId'].to_numpy(dtype=str)
        P_CALIB = pcalib.drop(columns = ['surveyId', 'speciesId']).to_numpy(dtype=np.float32)
        del pcalib

    else:
        pick = np.random.randint(0, len(sol), size = int(len(sol)*pval))
        tcalib = sol[pick]
        P_CALIB = PROBAS[pick]
        PROBAS = np.delete(PROBAS, pick, axis = 0)
        sol = np.delete(sol, pick, axis = 0)


    S = len(sol)
    ST = len(tcalib)
    N = len(PROBAS[0])

    SOL = np.zeros((S,N), dtype = np.intp)
    for i in range(S) :
        r_sol = sol[i].split(' ')
        for id in r_sol:
            SOL[i,int(id)] = 1

    T_CALIB = np.zeros((ST,N), dtype = np.intp)
    for i in range(ST) :
        t_p = tcalib[i].split(' ')
        for id in t_p:
            T_CALIB[i,int(id)] = 1
    del tcalib

    _, output = iterate(SOL, PROBAS, P_CALIB, T_CALIB)

    data_concatenated = [' '.join(map(str, row)) for row in output]
    p.DataFrame(
        {'surveyId': surveys,
        'speciesId': data_concatenated,
        }).to_csv("binary_predictions.csv", index = False)


if __name__ == "__main__":
    main()