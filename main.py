import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Una funzione che divide le età in fasce predefinite.
def dividi_eta(df, cut_points, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)
    return df


# Una funzione che crea una colonna per tipologia e assegna il valore 1 (e.g.: crea una colonna per fascia d'età e
# assegna 1 a quella di cui fa parte il passeggero, 0 alle altre).
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df


# Importa i CSV con i dati.
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# Mostra un grafico con i valori di sopravvivenza per classe di viaggio.
pclass_pivot = train.pivot_table(index="Pclass", values="Survived")
pclass_pivot.plot.bar()
plt.show()

# Mostra un grafico con i valori di sopravvivenza maschili e femminili.
sex_pivot = train.pivot_table(index="Sex", values="Survived")
sex_pivot.plot.bar()
plt.show()

# Definisce i punti di separazione delle categorie di età, arbitrari, e i nomi delle categorie.
fasce_eta = [-1, 0, 5, 12, 18, 35, 60, 100]
nomi_fasce = ["Sconosciuto", "Neonato", "Fanciullo", "Adolescente", "Giovane", "Uomo", "Vecchio"]

# Separa, ora, i valori, assegnando ad ogni passeggero come età solo la fascia di cui fa parte.
train = dividi_eta(train, fasce_eta, nomi_fasce)
test = dividi_eta(test, fasce_eta, nomi_fasce)

# Mostra un grafico con i valori di sopravvivenza per fascia d'età.
age_cat_pivot = train.pivot_table(index="Age_categories", values="Survived")
age_cat_pivot.plot.bar()
plt.show()

# Riformatta i dati in una tabella più semplice.
train = create_dummies(train, "Pclass")
test = create_dummies(test, "Pclass")
train = create_dummies(train, "Sex")
test = create_dummies(test, "Sex")
train = create_dummies(train, "Age_categories")
test = create_dummies(test, "Age_categories")

# Definisce le colonne della nuova tabella creata per essere presa in analisi.
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
           'Age_categories_Sconosciuto', 'Age_categories_Neonato',
           'Age_categories_Fanciullo', 'Age_categories_Adolescente',
           'Age_categories_Giovane', 'Age_categories_Uomo',
           'Age_categories_Vecchio']

# Crea il set intero di dati, ovvero un array di dati 0 e 1 (X) e un array di 'Survived' o meno (Y).
all_X = train[columns]
all_y = train['Survived']

# Crea un set ridotto di dati, e usa il rimanente per confrontare la precisione dell'insegnamento ricevuto.
train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.2, random_state=0)

# Insegna alla neural net.
# TODO: Implementare manualmente.
# L'indicazione manuale del solver serve per correggere un errore che non capisco.
lr = LogisticRegression(solver='lbfgs')
lr.fit(train_X, train_y)

# Predice i valori di test_X.
predictions = lr.predict(test_X)

# Confronta i valori con i risultati noti e definisce l'accuratezza.
accuracy = accuracy_score(test_y, predictions)
print(accuracy)
conf_matrix = confusion_matrix(test_y, predictions)
print(pd.DataFrame(conf_matrix, columns=['Survived', 'Died'], index=['P-Survived', 'P-Died']))

# Insegna ora all'AI usando tutti i dati a disposizione.
lr = LogisticRegression(solver='lbfgs')
lr.fit(all_X, all_y)
holdout = test
holdout_predictions = lr.predict(holdout[columns])

# Crea un CSV con i valori di predizione.
holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv('titanic_submission.csv', index=False)
