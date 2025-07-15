import numpy as np
import pandas as p

species_path = "~/Downloads/Donnees Sebastien/database_split.csv"
train_path = "~/Downloads/Donnees Sebastien 2/predictions_trainval.csv"
test_path = "~/Downloads/Donnees Sebastien 2/predictions_test.csv"

df_species = p.read_csv(species_path)
df_train = p.read_csv(train_path)
df_test = p.read_csv(test_path)

id_train = df_species[df_species["subset"] == "train"]["survey_id"]
id_test = df_species[df_species["subset"] == "test"]["survey_id"]

df_train = df_train[df_train["survey_id"].isin(id_train)]
df_test = df_test[df_test["survey_id"].isin(id_test)]

species_train = df_species[df_species["subset"] == "train"].iloc[:,29:-1]
species_test = df_species[df_species["subset"] == "test"].iloc[:,29:-1]


# pos = 421678
# neg = 40381264 - 421678
# a = neg / pos

print(np.shape(species_train))
print(np.shape(species_test))  
print(np.shape(df_train))
print(np.shape(df_test))

# df_train.iloc[:,1:] = df_train.iloc[:,1:] / (a + df_train.iloc[:,1:] * (1 - a))
# df_test.iloc[:,1:] = df_test.iloc[:,1:] / (a + df_test.iloc[:,1:] * (1 - a))

data_concatenated = []
for row in species_train.iterrows():
    specs = np.where(row[1] > 0)[0]
    data_concatenated.append(' '.join(map(str, specs)))

p.DataFrame(
    {'surveyId': id_train.to_numpy(dtype=str),
    'speciesId': data_concatenated,
    }).to_csv("rls_train_species.csv", index = False)

df_train.to_csv("rls_train_probas_ww.csv", index = False)

data_concatenated = []
for row in species_test.iterrows():
    specs = np.where(row[1] > 0)[0]
    data_concatenated.append(' '.join(map(str, specs)))


p.DataFrame(
    {'surveyId': id_test.to_numpy(dtype=str),
    'speciesId': data_concatenated,
    }).to_csv("rls_test_species.csv", index = False)

df_test.to_csv("rls_test_probas_ww.csv", index = False)