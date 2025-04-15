import numpy as np
import pandas as p


file = "data/hmsc_predictions.csv"
df = p.read_csv(file)

pred  = df.filter(regex=".*predict").copy()
sol = df.filter(regex=".*observed")
surveys = df["id_spygen"]


data_concatenated = []
for row in sol.iterrows():
    specs = np.where(row[1] == 1)[0]
    data_concatenated.append(' '.join(map(str, specs)))

p.DataFrame(
    {'surveyId': surveys.to_numpy(dtype=str),
    'speciesId': data_concatenated,
    }).to_csv("hmsc_test_species.csv", index = False)

pred.insert(0,'surveyId',surveys)
pred.to_csv("hmsc_test_probas.csv", index = False)
