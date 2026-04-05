import pandas as pd


def check_name_formatting(df: pd.DataFrame) -> str:
    if "Name" not in df.columns:
        return "La colonne Name est absente"

    is_ok = df["Name"].astype(str).str.contains(",").all()
    if is_ok:
        return "Test 'Name' OK se découpe toujours en 2 parties avec ','"
    return "Problème de format dans la colonne Name"


def check_missing_values(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return f"La colonne {column} est absente"

    missing_count = int(df[column].isna().sum())
    if missing_count == 0:
        return f"Pas de valeur manquante pour la variable {column}"
    return f"{missing_count} valeurs manquantes pour la variable {column}"


def check_data_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str = "PassengerId",
) -> str:
    if id_column not in train_df.columns or id_column not in test_df.columns:
        return f"La colonne {id_column} est absente dans train ou test"

    overlap = set(train_df[id_column]).intersection(set(test_df[id_column]))

    if len(overlap) == 0:
        return f"Pas de problème de data leakage pour la colonne {id_column}"
    return f"Problème de data leakage : {len(overlap)} identifiants communs entre train et test"