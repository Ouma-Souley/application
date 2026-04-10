import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import duckdb



from dotenv import load_dotenv
import os
import logging

from src.check import (
    check_name_formatting,
    check_missing_values
)

load_dotenv()
jeton_api = os.environ["JETON_API"]




titanic = pd.read_csv("data.csv")
con = duckdb.connect(database=":memory:")
def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler(),
        ],
    )
    
configure_logging()    
logging.info(check_name_formatting(titanic))
logging.info(check_missing_values(titanic, "Survived"))
logging.info(check_missing_values(titanic, "Age"))

# Check la structure de Name "Nom, Préno


parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_trees",
    type=int,
    default=20,
    help="Nombre d'arbres pour la random forest",
)
args = parser.parse_args()

n_trees = args.n_trees
max_depth = None
max_features = "sqrt"


## Encoder les données imputées ou transformées.
numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        (
            "Preprocessing categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=20)),
    ]
)


# splitting samples
y = titanic["Survived"]
x = titanic.drop("Survived", axis="columns")

# On _split_ notre _dataset_ d'apprentisage pour faire de la validation croisée une partie pour apprendre une partie pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



#jetonapi = "$trotskitueleski1917"

# Vérifie les valeurs manquantes





# Random Forest
("classifier", RandomForestClassifier(n_estimators=n_trees,  random_state=42)),

# Ici demandons d'avoir 20 arbres
pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)
# calculons le score sur le dataset d'apprentissage et sur le dataset de test (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction
rdmf_score = pipe.score(x_test, y_test)
rdmf_score_tr = pipe.score(x_train, y_train)
logging.info("%s de bonnes réponses sur les données de test pour validation", f"{rdmf_score:.1%}")
logging.info("--------------------")
logging.info("matrice de confusion")
logging.info("\n%s", confusion_matrix(y_test, y_pred))
