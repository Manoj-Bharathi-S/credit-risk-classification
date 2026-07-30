from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        random_state=1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def train_extra_trees(X_train, y_train):
    # These are the best params found in the notebook
    model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=1,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n{model_name} Performance :")
    print(f"Accuracy : {acc:.4f}")
    
    return {
        "model_name": model_name,
        "accuracy": acc
    }