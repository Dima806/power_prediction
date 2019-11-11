from model import PowerPrediction

power = PowerPrediction()

X = power.features.copy()
y = power.target.copy()

X, y = power.preprocess(X, y)

X = power.process_nans(X)

# TODO: work only for a single year, update later!
X['target'] = y
month_list = X.month.unique().tolist()
train_score_list = []
val_score_list = []
for last_month in month_list[1:]:
    df_train = X[X.month < last_month]
    df_val = X[X.month == last_month]

    y_train = df_train['target']
    X_train = df_train.drop(['target'], axis=1).copy()
    y_val = df_val['target']
    X_val = df_val.drop(['target'], axis=1).copy()
    power.fit(X_train, y_train)
    train_score = power.score(X_train, y_train)
    val_score = power.score(X_val, y_val)
    train_score_list.append(train_score)
    val_score_list.append(val_score)

mean_train_score = sum(train_score_list)/len(train_score_list)
mean_val_score = sum(val_score_list)/len(val_score_list)
print('>>> mean training score:', mean_train_score)
print('>>> mean validation score:', mean_val_score)
