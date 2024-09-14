from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    df = store.get_processed("transformed_dataset.csv")
    df_train, df_test = train_test_split(df, test_size=config["test_size"])

    grid_search = GridSearchCV(RandomForestClassifier(),param_grid=config["random_forest"],cv=5) 
    grid_search.fit(df_train[config["features"]], df_train[config["target"]]) 
    best_params = grid_search.best_estimator_
    print(best_params)

    rf_estimator = best_params#RandomForestClassifier(**best_params)
    model = SklearnClassifier(rf_estimator, config["features"], config["target"])
    model.train(df_train)

    metrics = model.evaluate(df_test)


    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)


if __name__ == "__main__":
    main()
