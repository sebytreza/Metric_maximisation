import numpy as np
import matplotlib.pyplot as plt
import pandas as p
import numba
from numba import cuda
from tqdm import tqdm
import math
import configparser


np.random.seed(1312)

## SCORE CALCULATION BASED ON SET VALUE METRICS ##
@numba.jit(fastmath=True, nogil=True, parallel = True)
def score(pred,target, func, args):
    VP = np.sum(np.logical_and(pred,target))
    FP = np.sum(pred) - VP
    FN = np.sum(target) - VP
    TN = len(pred) - VP - FP - FN
    return func(VP, FP, TN, FN, args)

## DEFINE OF UTILITY FUNCTIONS ##
@numba.njit(fastmath=True, nogil=True)
def f_b(TP, FP, TN, FN, args = [1]):
    b = args[0]
    return (1+b**2)*TP/((1+b**2)*TP + b*FN + FP)

@numba.njit(fastmath=True, nogil=True)
def jaccard(TP, FP, TN, FN, args = [1]):
    return TP/(TP + FP + FN)

## MAXIMIZATION OF THE SCORE ##
@numba.njit(fastmath=True, nogil=True, parallel = True)
def max_func_n3(p, func, args):
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
        U = 0.
        for i in range(K+1):
            for j in range(n - K + 1):
                U += C[K,i]*S[n-K,j]*func(i, K -i, n-K-j, j, args) #func(tp,fp,tn, fn)
            
        if U >= Umax :
            Umax, kmax  = U, K
        K += 1
    return kmax, Umax #return the number of species to keep and the score

@numba.njit(fastmath=True, nogil=True, parallel = True)
def max_f_b_n2(p, func, args):
    b = args[0]**(-2)
    p = np.sort(p)[::-1]
    n = len(p)
    Umax, kmax, U  = 0, 0, 0
    C = np.zeros((n+1,n+1))
    C[0,0] = 1
    for i in range(n): 
        C[i+1,0] = (1 - p[i])*C[i,0]
        for j in range(i+1):
            C[i+1,j+1] = p[i]*C[i,j] + (1 - p[i])*C[i,j+1]
              
    S = np.zeros(2*n)      
    for i in range(2*n):
        S[i] = 1/(i+1)
    
    k = 0
    while k<n and U == Umax:
        U = 0
        K = n - k
        for i in range(1, K+1):
            U += 2*i*C[K,i]*S[i + K -1]
        for i in range(2*(K-1)):
            S[i] = p[K-1]*S[i+1] + (1 - p[K-1])*S[i]
            
        if U >= Umax :
            Umax, kmax  = U, K
        k += 1
    return kmax, Umax
    
@numba.njit(fastmath=True, nogil=True, parallel = True)
def max_jaccard_n2(p, func, args):
    pass
    
## BASELINE STRATEGIES ##
@numba.njit(fastmath=True, nogil=True)
def threshold(p, th = 0.5) :
    K = 0
    while  K < len(p) and p[K] >= th:
        K += 1
    return K

@numba.njit(fastmath=True, nogil=True, parallel = True)
def sum_th(p) :
    return int(np.sum(p)) + 1


## APPLY DECISION FUNCTIONS TO EACH SITES ##
@numba.njit(fastmath=True, nogil = True)
def iterate(SOL, PROBAS, P_CALIB, T_CALIB, func, args, max_func):
    S = len(SOL)
    N = len(SOL[0])
    ST = len(T_CALIB)


    SCORE = np.zeros((S, 7))
    KLIST = np.zeros((S, 8))

    OUTPUT = []

    # Calibrate strategies on the given data #

    print('BEGIN CALIBRATION')
    # TopK
    Ktopk = 0
    Ut_topkmax = 0.
    Ktopkmax = 0

    upper_bound = 80

    upper_bound = min(upper_bound, N)
    if upper_bound > 100:
        print('WARNING: Topk calibration for many species may take a long time ...  \n' \
        'Consider changing the upper bound of the calibration')
    
    while Ktopk < upper_bound : #adapt upper bound if too many species
        Ut_topk = 0.
        for i in range(ST):
            probas = P_CALIB[i]
            tar = T_CALIB[i]
            sort = np.argsort(-probas)
            mask = np.where(probas > 0)[0]
            input = probas[sort][mask]
            topk = np.zeros(N)
            topk[sort[:Ktopk+1]] = 1
            Ut_topk += score(topk,tar,func, args)
        if Ut_topk >= Ut_topkmax:
            Ut_topkmax = Ut_topk
            Ktopkmax = Ktopk
        Ktopk += 1

    Ktopk = Ktopkmax +1
    print('Topk done')

    # Global threshold
    e = 0.01
    th = 1
    thmax = 1
    Uth_max = 0.
    while th >= 0:
        Ut_th = 0.
        for i in range(ST):
            probas = P_CALIB[i]
            tar = T_CALIB[i]
            sort = np.argsort(-probas)
            mask = np.where(probas > 0)[0]
            input = probas[sort][mask]
            Kth = threshold(input, th = th)
            nth = np.zeros(N)
            nth[sort[:Kth]] = 1
            Ut_th += score(nth,tar,func, args)
        if Ut_th >= Uth_max:
            Uth_max = Ut_th
            thmax = th
        th = th - e
    
    th = thmax
    print('Global threshold done')

    # Low threshold
    thlow = np.ones(N)
    for i in range(N):
        per = 100 - np.mean(T_CALIB[:,i])*100
        if per == 100 :
            thlow[i] = 1
        else:
            thlow[i] = np.percentile(P_CALIB[:,i], per)
    print('Percentile threshold done')

    # 5% threshold
    th5pc = np.ones(N)
    for i in range(N):
        occ = np.where(T_CALIB[:,i] == 1)[0]
        if len(occ) > 0:
            th5pc[i] = np.percentile(P_CALIB[occ,i], 5)
        else:
            th5pc[i] = 1
    print('95%Rec threshold done')


    print('CALIBRATION DONE')
    print('BEGIN BINARY PREDICTIONS')


    # Iterate throught the dataset #
    for i in numba.prange(S):

        probas = PROBAS[i]
        tar = SOL[i]
        
        sort = np.argsort(-probas)
        mask = np.where(input > 0.0)[0] #clip species with null probability
        input = probas[sort][mask]

        K, _ = max_func(input, func, args)
        pred = np.zeros(N).astype(np.intp)
        pred[sort[:K]] = 1
        U = score(pred,tar,func, args)

        
        topk = np.zeros(N)
        topk[sort[:Ktopk]] = 1
        U_topk = score(topk,tar,func, args)
    
        K0_5 = threshold(input, th = 0.5)
        n0_5 = np.zeros(N)
        n0_5[sort[:K0_5]] = 1
        U_0_5 = score(n0_5,tar,func, args)

        Kth = threshold(input, th)
        nth = np.zeros(N)
        nth[sort[:Kth]] = 1
        U_th = score(nth,tar,func, args)

        nper = probas[sort] > thlow
        Kper = np.sum(nper)
        U_per = score(nper,tar,func, args)

        n5pc = probas[sort] > th5pc
        K5pc = np.sum(n5pc)
        U_5pc = score(n5pc,tar,func, args)

        Ksum = sum_th(input)
        nsum = np.zeros(N)
        nsum[sort[:Ksum]] = 1
        U_sum = score(nsum,tar,func, args)

        KLIST[i] = np.array([K, Ktopk, K0_5, Kth, Kper, K5pc, Ksum, np.sum(tar)])
        SCORE[i] = np.array([U, U_topk, U_0_5, U_th, U_per, U_5pc, U_sum])


        OUTPUT.append(sort[:K])

    print("Maximization :" , np.mean(SCORE[:,0]))
    print("Topk         :" , np.mean(SCORE[:,1]))
    print("Th_0.5       :" , np.mean(SCORE[:,2]))
    print("Th_opti      :" , np.mean(SCORE[:,3]))
    print("Th_per       :" , np.mean(SCORE[:,4]))
    print("Th_95%Rec    :" , np.mean(SCORE[:,5]))
    print("Sum          :" , np.mean(SCORE[:,6]))

    return OUTPUT, KLIST,SCORE
 

def main():

    ## CONFIGURATION FILE ##
    config = configparser.ConfigParser()
    config.read('parameters.ini')

    ## DEFINE FILE PATH ##
    sol_file =  config['File paths']['sol_file']
    pred_file = config['File paths']['pred_file']
    tcalib_file = config['File paths']['tcalib_file']
    pcalib_file = config['File paths']['pcalib_file'] 

    ## DEFINE PARAMETERS OF EXPERIMENT ##
    pval = float(config['Experiment']['prob_val'])
    train_calib = bool(config['Experiment']['train_calib'])
    predict_val = bool(config['Experiment']['predict_val'])
    quad = bool(config['Experiment']['run_quad'])

    ## DEFINE UTILITY FUNCTION ##
    func = config['Utility function']['metric']
    param = config['Utility function']['param']

    if pval < 0 or pval > 1:
        raise ValueError("prob_val must be between 0 and 1")
    try :
        args = list(map(float, param.split(',')))
        if func == 'f_b':
            func = f_b
        elif func == 'jaccard':
            func = jaccard
        else:
            raise ValueError(f"{func} metric must be first implemented")
        _ = func(1, 1, 1, 1, args) 
    except ValueError:
         raise ValueError(f"{func} metric must be first implemented")
    
    ## DEFINE MAXIMIZATION FUNCTION ##

    if quad:
        if func == jaccard:
            max_func = max_jaccard_n2
        elif func == f_b and args[0] == int(args[0]):
            max_func = max_f_b_n2
        else:
            raise ValueError("Quadratic resolution impossible or not implemented")
    else:
        max_func = max_func_n3


    ## LOAD DATA ##
    sol = p.read_csv(sol_file)
    probas = p.read_csv(pred_file, delimiter = ',')
    probas = probas.join(sol.set_index('surveyId'), on='surveyId')
    probas = probas.dropna()
    surveys = probas['surveyId']
    sol = probas['speciesId'].to_numpy(dtype=str)
    PROBAS = probas.drop(columns = ['surveyId', 'speciesId']).to_numpy(dtype=np.float32)
    del probas
    
    pick = np.random.randint(0, len(sol), size = int(len(sol)*pval))

    if train_calib :
        tcalib  = p.read_csv(tcalib_file)
        pcalib  = p.read_csv(pcalib_file, delimiter = ',')
        pcalib = pcalib.join(tcalib.set_index('surveyId'), on='surveyId')
        pcalib = pcalib.dropna()
        tcalib = pcalib['speciesId'].to_numpy(dtype=str)
        P_CALIB = pcalib.drop(columns = ['surveyId', 'speciesId']).to_numpy(dtype=np.float32)
        del pcalib

    else:
        tcalib = sol[pick]
        P_CALIB = PROBAS[pick]
    
    if not predict_val:
        PROBAS = np.delete(PROBAS, pick, axis = 0)
        sol = np.delete(sol, pick, axis = 0)
        surveys = np.delete(surveys, pick, axis = 0)


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

    output, nb_species , score = iterate(SOL, PROBAS, P_CALIB, T_CALIB, func, args, max_func)

    data_concatenated = [' '.join(map(str, row)) for row in output]

    p.DataFrame(
        {'surveyId': surveys,
        'speciesId': data_concatenated,
        }).to_csv("submissions/binary_predictions.csv", index = False)
    
    p.DataFrame(
        score, 
        columns = ['max', 'topk', 'th_0.5', 'th_opti', 'th_per', 'th_5%', 'sum']
        ).to_csv("submissions/score_distrib.csv", index = False)

    p.DataFrame(
        nb_species, 
        columns = ['max', 'topk', 'th_0.5', 'th_opti', 'th_per', 'th_5%', 'sum', 'true']
        ).to_csv("submissions/nb_species.csv", index = False)


if __name__ == "__main__":
    main()