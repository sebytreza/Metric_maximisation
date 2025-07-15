import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as p
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score,auc, precision_recall_curve, roc_curve

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



def plot_aucs():
    sol_file = 'rls_train_species.csv'
    pred_file = 'rls_train_probas_ww.csv'

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


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    rec_grid = np.linspace(0.0, 1.0, 1000)
    mean_pr = np.zeros_like(rec_grid)

    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(S):
        if np.sum(SOL[i]) != 0 :
            pr, rec, _ = precision_recall_curve(SOL[i], PROBAS[i])
            mean_pr += np.interp(rec_grid, rec[::-1], pr[::-1])

            fpr,tpr,  _ = roc_curve(SOL[i], PROBAS[i])
            mean_tpr += np.interp(fpr_grid, fpr, tpr)

    mean_pr /= S
    mean_tpr /= S

    print("mean PR-AUC:", auc(rec_grid, mean_pr))
    print("mean ROC-AUC:", auc(fpr_grid, mean_tpr))

    ax1.plot(fpr_grid, mean_tpr, c = colors[0], linewidth = 2, label='Train, ROC-AUC = {:.2f}'.format(auc(fpr_grid, mean_tpr)))
    ax2.plot(rec_grid, mean_pr, c = colors[0], linewidth = 2, label='Train, PR-AUC = {:.2f}'.format(auc(rec_grid, mean_pr)))
    ax1.plot((0,1),(0,1), c = 'black', linewidth = 0.5)


    sol_file = 'rls_test_species.csv'
    pred_file = 'rls_test_probas_ww.csv'

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
    

    rec_grid = np.linspace(0.0, 1.0, 1000)
    mean_pr = np.zeros_like(rec_grid)

    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)


    for i in range(S):
        if np.sum(SOL[i]) != 0 :
            pr, rec, _ = precision_recall_curve(SOL[i], PROBAS[i])
            mean_pr += np.interp(rec_grid, rec[::-1], pr[::-1])

            fpr,tpr,  _ = roc_curve(SOL[i], PROBAS[i])
            mean_tpr += np.interp(fpr_grid, fpr, tpr)

    mean_pr /= S
    mean_tpr /= S

    print("mean PR-AUC:", auc(rec_grid, mean_pr))
    print("mean ROC-AUC:", auc(fpr_grid, mean_tpr))

    ax1.plot(fpr_grid, mean_tpr, c = colors[1], linewidth = 2, label='Test, ROC-AUC = {:.2f}'.format(auc(fpr_grid, mean_tpr)))
    ax2.plot(rec_grid, mean_pr, c = colors[1], linewidth = 2, label='Test, PR-AUC = {:.2f}'.format(auc(rec_grid, mean_pr)))

    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.grid()
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.set_title("PR Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.grid()
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def plot_score_distribution():
    title="Score Distribution"
    xlabel="Decision functions"
    ylabel="Score"
    score  = p.read_csv("submissions/score_distrib.csv")

    plt.figure(figsize=(10, 6))
    sns.violinplot(data= score, palette="muted")
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
    sns.violinplot(data= delta, palette="muted")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def plot_calibration_curve():
    title="Calibration Curve"
    xlabel="Predicted Probability"
    ylabel="Fraction of Positives"

    sol_file = 'rls_test_species.csv'
    pred_file = 'rls_test_probas_ww.csv'

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

    sol_file = 'data/GeoLifeCLEF_species.csv'
    pred_file = 'data/GeoLifeCLEF_probas.csv'


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
    Var = Y - np.sum(PROBAS**2, axis = 1)

    slope, intercept, r_value, _, _ = stats.linregress(X, Y)
    regression_line = slope * np.array([0,N]) + intercept

    plt.figure(figsize=(10, 10))
    # plt.plot((0,N),(0,N), c = 'gray', linewidth = 1)
    # plt.plot([0,N], regression_line, c= colors[0], alpha = 0.5, linewidth=2, label=f"Regression line (r={r_value:.2f})")
    # for  i in range(S):
    #     plt.plot([X[i], X[i]], [Y[i] - np.sqrt(20*Var[i]), Y[i] + np.sqrt(20*Var[i])], c = colors[0], alpha = 0.3, linewidth = 0.5)
    # plt.scatter(X,Y, c = colors[0], s = 10, alpha = 0.5 )
    X = np.log(X)/np.log(10)
    Y = np.log(np.round(Y,0))/np.log(10)
    B = max(max(X), max(Y))
    sns.scatterplot(x=X, y=Y, s=5, color=".15")
    sns.histplot(x=X, y=Y, bins=40, pthresh=.1, cmap="mako")
    sns.kdeplot(x=X, y=Y, levels=5, color="w", linewidths=1)
    sns.lineplot(x=[0, B], y=[0, B],color='red')
    plt.title(title, fontsize = 18)
    plt.xlabel(xlabel, fontsize = 14)
    plt.ylabel(ylabel, fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(0, B)
    plt.ylim(0, B)
    #plt.grid(linewidth = 1)
    #plt.xlim(0, N)
    #plt.ylim(0, N)
    plt.show()


def plot_pred_sr():
    title="Species richness"
    xlabel="True species richness"
    ylabel="Predicted species richness"

    # pred_file = 'submissions/predictions_rls_TSS.csv'
    # probas_file = 'data/rls_test_probas.csv'
    # spec_file = 'data/rls_test_species.csv'

    pred_file = 'submissions/predictions_GLC24_R2.csv'
    probas_file = 'data/GeoLifeCLEF_probas.csv'
    spec_file = 'data/GeoLifeCLEF_species.csv'


    pred_dt = p.read_csv(pred_file)
    probas = p.read_csv(probas_file)
    species = p.read_csv(spec_file)['speciesId'].to_numpy(dtype=str)

    probas = probas.join(pred_dt.set_index('surveyId'), on='surveyId')
    probas = probas.dropna()
    pred_dt = probas['speciesId'].to_numpy(dtype=str)

    PROBAS = probas.drop(columns = ['surveyId','speciesId']).to_numpy(dtype=np.float32)
    Y = []
    Xt = []
    S,N = PROBAS.shape
    for i in range(S) :
        Y.append(len(pred_dt[i].split(' ')))
        Xt.append(len(species[i].split(' ')))

    X = np.sum(PROBAS, axis = 1)

    Var = Y - np.sum(PROBAS**2, axis = 1)

    slope, intercept, r_value, _, _ = stats.linregress(X, Y)
    regression_line = slope * np.array([0,N]) + intercept

    plt.figure(figsize=(10, 10))
    X = np.log(X)/np.log(10)
    Y = np.log(np.round(Y,0))/np.log(10)
    B = max(max(X), max(Y))
    sns.scatterplot(x=X, y=Y, s=5, color=".15")
    sns.histplot(x=X, y=Y, bins=40, pthresh=.1, cmap="mako")
    sns.kdeplot(x=X, y=Y, levels=5, color="w", linewidths=1)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])
    plt.title(title, fontsize = 18)
    plt.xlabel(xlabel, fontsize = 14)
    plt.ylabel(ylabel, fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(0, B)
    plt.ylim(0, B)
    plt.grid(linewidth = 1)

    plt.figure(figsize=(10, 10))
    X = np.log(Xt)/np.log(10)
    B = max(max(X), max(Y))
    sns.scatterplot(x=X, y=Y, s=5, color=".15")
    sns.histplot(x=X, y=Y, bins=40, pthresh=.1, cmap="mako")
    sns.kdeplot(x=X, y=Y, levels=5, color="w", linewidths=1)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])
    plt.title(title, fontsize = 18)
    plt.xlabel(xlabel, fontsize = 14)
    plt.ylabel(ylabel, fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(0, B)
    plt.ylim(0, B)
    plt.grid(linewidth = 1)

    plt.show()





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


#run_permutation_test()
#plot_calibration_curve()
#plot_species_richness_all()
#plot_aucs()
#plot_species_richness()
plot_pred_sr()