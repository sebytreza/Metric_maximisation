import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as p
import seaborn as sb
from scipy import stats

colors = ['#11999E', '#40514E', '#FFB22C']

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
    for i in range(0, len(score.columns)-1):
        res = stats.permutation_test((score.iloc[:,-1], score.iloc[:,i]), statistic = statistic, vectorized=True, alternative='greater')
        print(f"{names[-1]} vs {names[i]}: S = {res.statistic}, p-value = {res.pvalue}")

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
    delta = nb_species - nb_species['True'].values[:, np.newaxis]
    delta = delta.drop(columns = ['True'])

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

    sol_file = 'data/birds_study_species.csv'
    pred_file = 'data/birds_study_calib.csv'

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
    for bin in bins:
        idx = np.where((PROBAS >= bin)*(PROBAS < bin+dp)) 
        if len(idx[0]!= 0) :
            Y.append(np.mean(SOL[idx]))
            sY.append(np.std(SOL[idx])/np.sqrt(len(idx[0])))
            X.append(np.mean(PROBAS[idx]))
    X, Y, sY = np.array(X), np.array(Y), np.array(sY)
    plt.figure(figsize=(10, 6))
    #plt.plot(X,d_freq, c = 'cyan', linewidth = 0.5)
    plt.scatter(X,Y, c = colors[0], s = 50 )
    plt.plot(X,Y, c = colors[0] , linewidth = 1)
    plt.plot(X,Y + 3*sY,'--',c = colors[0] , linewidth = 0.5)
    plt.plot(X,Y - 3*sY,'--',c = colors[0] , linewidth = 0.5)

    plt.plot((0,1),(0,1), c = 'black', linewidth = 0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.fill_between(X, Y + 3*sY, Y - 3*sY, color= colors[0], alpha=0.1)
    plt.gca().set_aspect('equal')
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


def plot_species_richness():
    title="Species richness"
    xlabel="True species richness"
    ylabel="Predicted species richness"

    sol_file = 'data/birds_study_species.csv'
    pred_file = 'data/birds_study_calib.csv'

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

    X = np.sum(SOL, axis = 1)
    Y = np.sum(PROBAS, axis = 1)

    slope, intercept, r_value, _, _ = stats.linregress(X, Y)
    regression_line = slope * np.array([0,N]) + intercept

    plt.figure(figsize=(10, 6))
    plt.plot((0,N),(0,N), c = 'gray', linewidth = 1)
    plt.plot([0,N], regression_line, c= colors[0], alpha = 0.5, linewidth=2, label=f"Regression line (r={r_value:.2f})")
    plt.scatter(X,Y, c = colors[0], s = 100, alpha = 0.5 )
    plt.title(title, fontsize = 18)
    plt.xlabel(xlabel, fontsize = 14)
    plt.ylabel(ylabel, fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.grid(linewidth = 1)
    plt.xlim(0, N)
    plt.ylim(0, N)
    plt.show()

def plot_species_range():
    title="Species range"
    xlabel="True species range"
    ylabel="Predicted species range"

    sol_file = 'data/birds_study_species.csv'
    pred_file = 'data/birds_study_calib.csv'

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

    X = np.sum(SOL, axis = 0)
    Y = np.sum(PROBAS, axis = 0)

    slope, intercept, r_value, _, _ = stats.linregress(X, Y)
    regression_line = slope * np.array([0,S]) + intercept

    plt.figure(figsize=(10, 6))
    plt.plot((0,S),(0,S), c = 'gray', linewidth = 1)
    plt.plot([0,S], regression_line, c= colors[0], alpha = 0.5, linewidth=2, label=f"Regression line (r={r_value:.2f})")
    plt.scatter(X,Y, c = colors[0], s = 100, alpha = 0.5 )
    plt.title(title, fontsize = 18)
    plt.xlabel(xlabel, fontsize = 14)
    plt.ylabel(ylabel, fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.grid(linewidth = 1)
    plt.xlim(0, S)
    plt.ylim(0, S)
    plt.show()


def plot_species_richness_all():
    title="Species richness"
    xlabel="True species richness"
    ylabel="Predicted species richness"

    nb_species = p.read_csv("submissions/nb_species.csv")
    true_nb_species = nb_species['True'].values
    prob_nb_species = nb_species['Sum'].values
    pred_nb_species = nb_species.drop(columns = ['True', 'Sum']).to_numpy(dtype=np.float32)
    N = max(pred_nb_species.max(), true_nb_species.max())

    slopepb, interceptpb, r_valuepb, _, _ = stats.linregress(true_nb_species, prob_nb_species)
    regression_linepb = slopepb * np.array([0,N]) + interceptpb

    X = true_nb_species
    Ypb = prob_nb_species

    for i in range(pred_nb_species.shape[1]):
        plt.figure(figsize=(10, 6))
        plt.plot((0,N),(0,N), c = 'gray', linewidth = 1)
        plt.grid(linewidth = 1)


        Y = pred_nb_species[:,i]
        slope, intercept, r_value, _, _ = stats.linregress(X, Y)
        regression_line = slope * np.array([0,N]) + intercept

        # plt.plot([0,N], regression_linepb, c= colors[0],  alpha = 1, linewidth=2, label=f"Regression line (r={r_valuepb:.2f})")
        # plt.scatter(X,Ypb, c = colors[0], s = 100, alpha = 0.5, marker = 's',label="SSE species richness")

        # plt.plot([0,N], regression_line, c= colors[2], alpha = 1, linewidth=2, label=f"Regression line (r={r_value:.2f})")
        # plt.scatter(X,Y, c = colors[2], s = 100, alpha = 0.5, marker = 's')
        plt.hist2d(X, Y, bins= [np.arange(0, N, 1), np.arange(0, N, 1)], cmap= mcolors.LinearSegmentedColormap.from_list("cgrad", [colors[0]+"20", colors[0]] ) , cmin=1)
        plt.title(title + f" ({nb_species.columns[i]})", fontsize = 18)
        plt.xlabel(xlabel, fontsize = 14)
        plt.ylabel(ylabel, fontsize = 14)
        plt.gca().set_aspect('equal')
        plt.xlim(0, N)
        plt.ylim(0, N)
    plt.show()




plot_species_range()