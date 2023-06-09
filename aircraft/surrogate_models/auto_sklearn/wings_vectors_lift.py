
import pandas as pd
from sklearn.model_selection import train_test_split

import autosklearn.regression
import matplotlib.pyplot as plt
from autosklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score


print(autosklearn.__version__)


if __name__ == '__main__':
    df = pd.read_csv('../wings_vectors_drags_lifts.csv')

    which_data = 'lift'

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['i', 'name', which_data], axis=1), df[which_data], test_size=0.2, random_state=777)


    y_test.head()


    automl = autosklearn.regression.AutoSklearnRegressor(
        per_run_time_limit=1080, #1080
        time_left_for_this_task=10800, #10800
        metric=root_mean_squared_error,
        memory_limit=3072*11,
        n_jobs=-1)
    automl.fit(X_train, y_train)
    automl.get_params()

    print(automl.sprint_statistics())


    # evaluate best model
    y_hat = automl.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_hat)
    print("RMSE: %.3f" % rmse)
    mae = mean_absolute_error(y_test, y_hat)
    print("MAE: %.3f" % mae)
    r_2 = r2_score(y_test, y_hat)
    print("r^2: %.3f" % r_2)


    import numpy as np
    y_train_hat = automl.predict(X_train)
    #save np array y_train_hat to a csv file
    np.savetxt(f'./wings_vectors_y_train_hat_{which_data}.csv', y_train_hat, delimiter=',')
    np.savetxt(f'./wings_vectors_y_test_hat_{which_data}.csv', y_hat, delimiter=',')


    model_leadership = automl.leaderboard(detailed = True, ensemble_only=True, sort_order="descending")
    model_leadership

    #save the model
    import pickle
    import os

    save_path = f'./wings/{which_data}_model' 
    model_file_name = 'final_model.sav'
    save_file = save_path + '/' + model_file_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pickle.dump(automl, open(save_file, 'wb'))


    #load the model
    loaded_model = pickle.load(open(save_file, 'rb'))
    # evaluate best model
    y_hat = loaded_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_hat)
    print("RMSE: %.3f" % rmse)
    mae = mean_absolute_error(y_test, y_hat)
    print("MAE: %.3f" % mae)
    r_2 = r2_score(y_test, y_hat)
    print("r^2: %.3f" % r_2)

