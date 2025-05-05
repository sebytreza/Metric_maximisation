import matplotlib.pyplot as plt
import numpy as np
import pandas as p
import seaborn as sb
from scipy import stats



def run_wmw_test():
    score = p.read_csv("submissions/score_distrib.csv")
    names = score.columns
    for i in range(1, len(score.columns)):
        wmw, pvalue = stats.mannwhitneyu(score.iloc[:,0], score.iloc[:,i], alternative='greater')
        print(f"{names[0]} vs {names[i]}: WMW = {wmw}, p-value = {pvalue}")

def statistic(x, y, axis):
    return np.mean(x, axis) - np.mean(y, axis)

def run_permutation_test():
    score = p.read_csv("submissions/score_distrib.csv")
    names = score.columns
    for i in range(1, len(score.columns)):
        res = stats.permutation_test((score.iloc[:,0], score.iloc[:,i]), statistic = statistic, vectorized=True, alternative='greater')
        print(f"{names[0]} vs {names[i]}: S = {res.statistic}, p-value = {res.pvalue}")

def plot_score_distribution():
    title="Score Distribution"
    xlabel="Decision functions"
    ylabel="Score"
    score  = p.read_csv("submissions/score_distrib.csv")

    plt.figure(figsize=(10, 6))
    sb.violinplot(data= score, palette="muted")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def plot_nb_species():
    title="Number of Species Difference"
    xlabel="Decision functions"
    ylabel="Delta from True"

    nb_species = p.read_csv("submissions/nb_species.csv")
    delta = nb_species - nb_species['true'].values[:, np.newaxis]
    delta = delta.drop(columns = ['true'])

    plt.figure(figsize=(10, 6))
    sb.violinplot(data= delta, palette="muted")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def plot_calibration_curve():
    title="Calibration Curve"
    xlabel="Predicted Probability"
    ylabel="Fraction of Positives"

    sol_file = "data_examples/hmsc_test_species.csv"
    pred_file = "data_examples/hmsc_test_probas.csv"

    sol_file = 'data/birds_study_species.csv'
    pred_file = 'data/birds_study_probas.csv'

    sol = p.read_csv(sol_file)
    probas = p.read_csv(pred_file)
    probas = probas.join(sol.set_index('surveyId'), on='surveyId')
    probas = probas.dropna()
    sol = probas['speciesId'].to_numpy(dtype=str)

    PROBAS = probas.drop(columns = ['surveyId', 'speciesId']).to_numpy(dtype=np.float32)

    S,N = PROBAS.shape
    SOL = np.zeros((S,N), dtype = np.intp)
    for i in range(S) :
        r_sol = sol[i].split(' ')
        for id in r_sol:
            SOL[i,int(id)] = 1

    dp = 0.05
    bins = np.arange(0,1,dp)
    Y = []
    X = []
    sY = []
    sp_freq = [SOL[:,i].mean() for i in range(N)]
    d_freq = np.zeros_like(bins)
    for bin in bins:
        idx = np.where((PROBAS >= bin)*(PROBAS < bin+dp))
        for id in idx:
            d_freq[int(bin/dp)] += sp_freq[id[1]]
        d_freq[int(bin/dp)] /= len(idx)
            
        if len(idx[0]!= 0) :
            Y.append(np.mean(SOL[idx]))
            sY.append(np.std(SOL[idx])/np.sqrt(len(idx[0])))
            X.append(np.mean(PROBAS[idx]))
    X, Y, sY = np.array(X), np.array(Y), np.array(sY)
    plt.figure(figsize=(10, 6))
    plt.plot(X,d_freq, c = 'cyan', linewidth = 0.5)
    plt.scatter(X,Y, c = 'orange')
    plt.plot(X,Y, c = 'orange', linewidth = 1)
    plt.plot(X,Y + 3*sY,'--',c = 'orange', linewidth = 0.5)
    plt.plot(X,Y - 3*sY,'--',c = 'orange', linewidth = 0.5)

    plt.plot((0,1),(0,1), c = 'black', linewidth = 0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.fill_between(X, Y + 3*sY, Y - 3*sY, color='orange', alpha=0.1)
    plt.gca().set_aspect('equal')
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

run_permutation_test()
