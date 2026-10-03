# setup.py
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("UCI_Credit_Card.csv")
# print(df.shape)  # (30000, 25)
# print(df["default.payment.next.month"].mean().round(3))  # ~0.221 default rate

# section1_prep.py
df = df.rename(columns={"default.payment.next.month": "default", "PAY_0": "PAY_1"})
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})  # undocumented -> "other"
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})  # undocumented -> "other"
df = df.drop(columns=["ID"])  # an identifier, not a feature

categorical = ["SEX", "EDUCATION", "MARRIAGE"]
numeric = (
    ["LIMIT_BAL", "AGE"]
    + [f"PAY_{i}" for i in range(1, 7)]  # repayment status, 6 months
    + [f"BILL_AMT{i}" for i in range(1, 7)]  # bill amounts
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)  # payment amounts

# section1_pipeline.py


def build_prep(numeric_cols, categorical_cols):
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


# section2_regression.py

num_r = (
    ["AGE"]
    + [f"PAY_{i}" for i in range(1, 7)]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
Xr, yr = df[num_r + categorical], df["LIMIT_BAL"]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25, random_state=0)

for name, reg in [
    ("linear", LinearRegression()),
    ("ridge", Ridge(alpha=1.0)),
    ("lasso", Lasso(alpha=1.0)),
]:
    model = Pipeline([("prep", build_prep(num_r, categorical)), ("reg", reg)]).fit(
        Xr_tr, yr_tr
    )
    pred = model.predict(Xr_te)
    print(
        f"{name:7s}  R2 = {r2_score(yr_te, pred):.3f}   "
        f"RMSE = {np.sqrt(mean_squared_error(yr_te, pred)):,.0f}"
    )

model = Pipeline(
    [("prep", build_prep(num_r, categorical)), ("reg", LinearRegression())]
).fit(Xr_tr, yr_tr)
resid = yr_te - model.predict(Xr_te)


plt.scatter(model.predict(Xr_te), resid, alpha=0.3, s=8)
plt.axhline(0, color="red")
plt.xlabel("predicted LIMIT_BAL")
plt.ylabel("residual")
plt.show()

# section3_classification.py

num_c = (
    ["LIMIT_BAL", "AGE"]
    + [f"PAY_{i}" for i in range(1, 7)]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
Xc, yc = df[num_c + categorical], df["default"]
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.25, random_state=0, stratify=yc
)

models = {
    "logistic": LogisticRegression(max_iter=2000),
    "tree": DecisionTreeClassifier(max_depth=4, random_state=0),
    "forest": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
    "histgb": HistGradientBoostingClassifier(random_state=0),
}
for name, clf in models.items():
    m = Pipeline([("prep", build_prep(num_c, categorical)), ("clf", clf)]).fit(
        Xc_tr, yc_tr
    )
    p = m.predict(Xc_te)
    pp = m.predict_proba(Xc_te)[:, 1]
    print(
        f"{name:9s} acc={accuracy_score(yc_te, p):.3f} prec={precision_score(yc_te, p):.3f} "
        f"rec={recall_score(yc_te, p):.3f} F1={f1_score(yc_te, p):.3f} AUC={roc_auc_score(yc_te, pp):.3f}"
    )

# section4_learning_curve.py

est = Pipeline(
    [
        ("prep", build_prep(num_c, categorical)),
        ("clf", DecisionTreeClassifier(max_depth=None, random_state=0)),
    ]
)
sizes, train, val = learning_curve(
    est,
    Xc,
    yc,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 6),
    scoring="roc_auc",
    n_jobs=-1,
)
plt.plot(sizes, train.mean(1), marker="o", label="training")
plt.plot(sizes, val.mean(1), marker="o", label="validation")
plt.xlabel("training-set size")
plt.ylabel("ROC-AUC")
plt.legend()
plt.show()

# section4_validation_curve.py

depths = [1, 2, 3, 4, 6, 8, 12, None]
train, val = validation_curve(
    Pipeline(
        [
            ("prep", build_prep(num_c, categorical)),
            ("clf", DecisionTreeClassifier(random_state=0)),
        ]
    ),
    Xc,
    yc,
    param_name="clf__max_depth",
    param_range=depths,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)

for d, v in zip(depths, val.mean(1)):
    print(f"max_depth={str(d):>4}  validation AUC={v:.3f}")

# section5_crossval.py

pipe = Pipeline(
    [
        ("prep", build_prep(num_c, categorical)),
        ("clf", LogisticRegression(max_iter=2000)),
    ]
)
scores = cross_val_score(pipe, Xc, yc, cv=5, scoring="roc_auc")
print(
    scores.round(3), " mean =", scores.mean().round(3), " std =", scores.std().round(3)
)

# section5_accuracy_trap.py
print("model accuracy (logistic):", 0.812)
print("'always predict no-default' accuracy:", round(1 - yc_te.mean(), 3))  # ~0.779

logit = pipe.fit(Xc_tr, yc_tr)
cm = confusion_matrix(yc_te, logit.predict(Xc_te))
print(cm)  # [[TN, FP], [FN, TP]]
ConfusionMatrixDisplay(cm, display_labels=["no default", "default"]).plot()
plt.show()

# section5_curves.py
pp = logit.predict_proba(Xc_te)[:, 1]
RocCurveDisplay.from_predictions(yc_te, pp)
plt.show()
PrecisionRecallDisplay.from_predictions(yc_te, pp)
plt.show()
# section5_threshold.py
for t in [0.5, 0.3, 0.2]:
    pred = (pp >= t).astype(int)
    print(
        f"threshold {t}:  recall = {recall_score(yc_te, pred):.3f}  "
        f"precision = {precision_score(yc_te, pred):.3f}"
    )
