"""Feature selection by iterative Random Forest elimination.

Idea: train a Random Forest, look at which feature it uses the least, remove it, train again.
Repeat until few features are left. Then plot the validation error against the number of
features: the error stays flat while we remove useless features, and rises when we start
removing useful ones. The smallest set before the error rises is our selection.

Rules (see docs/feature_selection_plan.md):
- Only the training part of the data is used (split into "fit" and "validation").
  The test set is never seen here.
- Preprocessing (Imputer, StringIndexer) is fit once, on the fit part only.
- Categorical columns are indexed but NOT one-hot encoded: Random Forest understands
  indexed categories, so each categorical column counts as ONE feature.

Usage from a notebook in notebooks/:
    from src.feature_selection import split_for_selection, rf_elimination, choose_features
    fit_df, valid_df = split_for_selection(df_clean)
    rounds = rf_elimination(fit_df, valid_df, num_cols, cat_cols)
    final_features = choose_features(rounds)
"""
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import DataFrame, functions as F
import pandas as pd


def split_for_selection(df: DataFrame):
    """Split by listing id: 60% fit, 20% validation, 20% test (thrown away).

    hash(id) % 10 gives each listing a fixed number from 0 to 9, so a listing always lands
    in the same part, whatever the DataFrame, its columns or its partitions.
    (randomSplit does not guarantee this: it depends on how Spark partitions the data.)

      bucket 0-5 -> fit         (the RF learns here)
      bucket 6-7 -> validation  (we measure the error here)
      bucket 8-9 -> test        (same rule as the main pipeline; never used here)
    """
    bucket = F.abs(F.hash("id")) % 10
    fit_df = df.filter(bucket < 6)
    valid_df = df.filter((bucket >= 6) & (bucket < 8))
    return fit_df, valid_df


def _prepare(fit_df: DataFrame, valid_df: DataFrame, num_cols, cat_cols):
    """Fill nulls and index categories. Fit on fit_df only, apply to both parts.

    Returns the two transformed DataFrames and the name of the prepared column for each
    original feature, e.g. {"beds": "beds_imp", "room_type": "room_type_idx"}.
    """
    imputer = Imputer(strategy="median",
                      inputCols=num_cols,
                      outputCols=[c + "_imp" for c in num_cols])
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
                for c in cat_cols]

    prep_model = Pipeline(stages=[imputer] + indexers).fit(fit_df)

    prepared_name = {c: c + "_imp" for c in num_cols}
    prepared_name.update({c: c + "_idx" for c in cat_cols})

    fit_ready = prep_model.transform(fit_df).cache()
    valid_ready = prep_model.transform(valid_df).cache()
    return fit_ready, valid_ready, prepared_name


def _train_and_score(fit_ready, valid_ready, features, prepared_name, seed, num_trees, max_depth):
    """Train one Random Forest on `features`. Returns (importance per feature, RMSE, R²)."""
    assembler = VectorAssembler(inputCols=[prepared_name[f] for f in features],
                                outputCol="features")
    rf = RandomForestRegressor(featuresCol="features", labelCol="label",
                               numTrees=num_trees, maxDepth=max_depth,
                               maxBins=64,  # must be >= number of categories (20 arrondissements)
                               seed=seed)

    model = rf.fit(assembler.transform(fit_ready))
    predictions = model.transform(assembler.transform(valid_ready))

    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

    # featureImportances is a vector in the same order as the assembler's input columns.
    importances = dict(zip(features, model.featureImportances.toArray()))
    return importances, rmse, r2


def rf_elimination(fit_df: DataFrame, valid_df: DataFrame, num_cols, cat_cols,
                   seeds=(42, 43, 44), num_trees=100, max_depth=10, min_features=2):
    """Iterative Random Forest elimination.

    Each round: train one RF per seed, average the importances and the scores,
    record the round, then remove the least important feature(s):
    3 per round while more than 20 features are left, then 1 per round.

    Returns a pandas DataFrame with one row per round (small table, fine for pandas).
    """
    fit_ready, valid_ready, prepared_name = _prepare(fit_df, valid_df, num_cols, cat_cols)

    features = list(num_cols) + list(cat_cols)
    rounds = []

    while True:
        # Train one forest per seed and average, so a feature is not removed by bad luck.
        importance_sum = {f: 0.0 for f in features}
        rmse_sum, r2_sum = 0.0, 0.0
        for seed in seeds:
            importances, rmse, r2 = _train_and_score(fit_ready, valid_ready, features,
                                                     prepared_name, seed, num_trees, max_depth)
            for f in features:
                importance_sum[f] += importances[f]
            rmse_sum += rmse
            r2_sum += r2

        avg_importance = {f: importance_sum[f] / len(seeds) for f in features}
        avg_rmse = rmse_sum / len(seeds)
        avg_r2 = r2_sum / len(seeds)

        # Sort from least to most important.
        ranking = sorted(features, key=lambda f: avg_importance[f])

        if len(features) <= min_features:
            removed = []
        else:
            n_remove = 3 if len(features) > 20 else 1
            n_remove = min(n_remove, len(features) - min_features)
            removed = ranking[:n_remove]

        rounds.append({
            "n_features": len(features),
            "rmse_log": avg_rmse,
            "r2": avg_r2,
            "features": list(features),
            "removed_next": removed,
            "importances": {f: round(float(avg_importance[f]), 4) for f in ranking[::-1]},
        })
        print(f"{len(features):>3} features | RMSE (log) = {avg_rmse:.4f} | R² = {avg_r2:.3f}"
              f" | removing: {removed}")

        if not removed:
            break
        features = [f for f in features if f not in removed]

    fit_ready.unpersist()
    valid_ready.unpersist()
    return pd.DataFrame(rounds)


def choose_features(rounds: pd.DataFrame, tolerance=0.01):
    """Selection rule, fixed before running: the smallest feature set whose validation RMSE
    is within `tolerance` (1%) of the best round."""
    best_rmse = rounds["rmse_log"].min()
    good_enough = rounds[rounds["rmse_log"] <= best_rmse * (1 + tolerance)]
    chosen = good_enough.sort_values("n_features").iloc[0]
    return chosen["features"]
