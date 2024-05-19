from catboost import (
    CatBoostClassifier,
    EFeaturesSelectionAlgorithm,
    EShapCalcType,
    Pool,
)
from feature_engine.selection import DropCorrelatedFeatures

from exploration.models.otk import constants


def select_features_syntetic(data, features, steps: int = 1):
    algorithm = EFeaturesSelectionAlgorithm.RecursiveByPredictionValuesChange
    train_pool = Pool(data[features], data[constants.DEFECT_COL])
    model = CatBoostClassifier(iterations=1000, random_seed=0)
    summary = model.select_features(
        train_pool,
        features_for_select=list(range(data[features].shape[1])),
        num_features_to_select=20,
        steps=steps,    # more steps - more accurate selection
        algorithm=algorithm,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=True,    # to train model with selected features
        logging_level='Silent',
        plot=False
    )
    return summary


def get_low_corr_features(data, features):
    dcf = DropCorrelatedFeatures(threshold=0.7, )
    return list(
        dcf.fit_transform(X=data[features],
                          y=data[constants.DEFECT_COL]).columns
    )
