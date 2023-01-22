import pandas as pd
import numpy as np
import json
import math
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime, timedelta, date

def forecast_volume():

    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    df = pd.read_pickle('./source/SO_date_meta_cat_ewm.pkl') 

    with open('./source/date_params.json', 'r') as f:
        date_json = json.load(f)

    date_params = date_json[0]

    df.set_index("delivery_date", inplace = True)

    train_df = df.loc[date_params['train_date_start']:date_params['train_date_end']]

    print(train_df.info())

    #Weekend Indicator as Regressor

    train_df['Weekend_Indicator'] = [None] * len(train_df['day_of_week'])
    for i in range(len(train_df['day_of_week'])):
        if (train_df['day_of_week'][i] == 5) or (train_df['day_of_week'][i] == 6):
            train_df['Weekend_Indicator'][i] = 1
        else:
            train_df['Weekend_Indicator'][i] = 0

    #Generating Test Set

    test_dict = {
    "delivery_date" : [], 
    "plant" : [],
    "parent_name" : [],
    "distribution_channel" : [],
    "y_ewm_7" : [],
    "y_ewm_14" : [],
    }

    """y_ewm_7" : [],
    "y_ewm_14" : [],
    "y_lag_neg_3": [],
    "y_7_pct_change": [],
    "y_rol_lag_1": [],"""

    for dates in list(pd.date_range(start = date_params['test_date_start'], end = date_params['test_date_end'], freq = 'D')):
        for i in train_df['plant'].unique():
            for j in train_df['distribution_channel'].unique():
                for k in train_df['parent_name'].unique():
                    
                    test_dict['delivery_date'].append(dates)
                    test_dict['plant'].append(i)
                    test_dict['distribution_channel'].append(j)
                    test_dict['parent_name'].append(k)
                    test_dict['y_ewm_7'].append(0)
                    test_dict['y_ewm_14'].append(0)
                    """test_dict["y_lag_neg_3"].append(0)
                    test_dict["y_7_pct_change"].append(0)
                    test_dict["y_rol_lag_1"].append(0)"""

    test_df = pd.DataFrame(test_dict)

    #Adding Date Metadata and Weekend Indicator Regressor to the Test Set

    def week_of_month(tgtdate):
        import calendar
        tgtdate = pd.to_datetime(tgtdate)
        days_this_month = calendar.mdays[tgtdate.month]
        for i in range(1, days_this_month):
            d = datetime(tgtdate.year, tgtdate.month, i)
            if d.day - d.weekday() > 0:
                startdate = d
                break
        # now we canuse the modulo 7 appraoch
        return (tgtdate - startdate).days //7 + 1

    test_df['delivery_date'] = pd.to_datetime(test_df['delivery_date'])
    test_df['day_of_week'] = test_df['delivery_date'].dt.dayofweek 
    #test_df['day_of_month'] = test_df['delivery_date'].dt.day
    test_df['week_of_month'] = test_df['delivery_date'].apply(week_of_month)
    test_df['Weekend_Indicator'] = [None] * len(test_df['day_of_week'])
    for i in range(len(test_df['delivery_date'])):
        if (test_df['day_of_week'][i] == 5) or (test_df['day_of_week'][i] == 6):
            test_df['Weekend_Indicator'][i] = 1
        else:
            test_df['Weekend_Indicator'][i] = 0

    #Multi-Column Label Encoding Method
    class MultiColumnLabelEncoder:

        def __init__(self, columns=None):
            self.columns = columns # array of column names to encode

        
        def fit(self, X, y=None):
            self.encoders = {}
            columns = X.columns if self.columns is None else self.columns
            for col in columns:
                self.encoders[col] = LabelEncoder().fit(X[col])
            return self


        def transform(self, X):
            output = X.copy()
            columns = X.columns if self.columns is None else self.columns
            for col in columns:
                output[col] = self.encoders[col].transform(X[col])
            return output


        def fit_transform(self, X, y=None):
            return self.fit(X,y).transform(X)


        def inverse_transform(self, X):
            output = X.copy()
            columns = X.columns if self.columns is None else self.columns
            for col in columns:
                output[col] = self.encoders[col].inverse_transform(X[col])
            return output

    test_df.set_index('delivery_date', inplace = True)
    train_test_df = pd.concat([train_df, test_df], axis = 0)
    train_test_df.drop(['plant_name', 'parent_code', 'parent_uom'], axis = 1, inplace = True)
    #train_test_df['Weekend_Indicator'] = train_test_df['Weekend_Indicator'].astype(int)

    def _split_train_test_bydate(dataframe, index_col, start_date, end_date, target_var):  
        from datetime import datetime, timedelta
        dframe = dataframe
        try:
            if target_var == 'CQ_Dplus2':
                dframe = dframe.drop('CQ_Dplus1', axis = 1)
            elif target_var == 'CQ_Dplus1':
                dframe = dframe.drop('CQ_Dplus2', axis = 1)
            dframe = dframe.reset_index()
            dframe = dframe.set_index(index_col)
            dframe = dframe.sort_index(ascending = True)
            X, y = dframe.loc[ : , dframe.columns != target_var ], dframe[target_var]
            train_x, train_y = X.loc[:pd.to_datetime(start_date) - timedelta(1)], y.loc[:pd.to_datetime(start_date) - timedelta(1)] 
            test_x, test_y = X.loc[(pd.to_datetime(start_date)):end_date], y.loc[(pd.to_datetime(start_date)):end_date]
            mle = MultiColumnLabelEncoder(columns = ['parent_name']) # 'day_of_week', 'Weekend_Indicator', 'week_of_year'
            train_x = mle.fit_transform(train_x)
            train_x_inv = mle.inverse_transform(train_x)
            test_x = mle.fit_transform(test_x)
            test_x_inv = mle.inverse_transform(test_x)
        except: 
            dframe = dframe.reset_index()
            dframe = dframe.set_index(index_col)
            dframe = dframe.sort_index(ascending = True)
            X, y = dframe.loc[ : , dframe.columns != target_var ], dframe[target_var]
            train_x, train_y = X.loc[:pd.to_datetime(start_date) - timedelta(1)], y.loc[:pd.to_datetime(start_date) - timedelta(1)]
            test_x, test_y = X.loc[(pd.to_datetime(start_date)):end_date], y.loc[(pd.to_datetime(start_date)):end_date]
            mle = MultiColumnLabelEncoder(columns = ['parent_name']) #'day_of_week', 'Weekend_Indicator', 'week_of_year'
            train_x = mle.fit_transform(train_x)
            train_x_inv = mle.inverse_transform(train_x)
            test_x = mle.fit_transform(test_x)
            test_x_inv = mle.inverse_transform(test_x)
        return train_x, train_y, test_x, test_y, train_x_inv, test_x_inv

    np.random.seed(42)

    def _model_arima(trainx, trainy, testx, testy):
        from pmdarima import auto_arima
        aarima_model = auto_arima(trainy, X = trainx, information_criterion = 'aic', start_p=2, d=0, start_q=2, max_p=2, max_d=2, max_q=7, suppress_warnings=True, error_action='ignore', 
                                alpha=0.05, test='kpss', seasonal_test='ch', stepwise=True, n_jobs=1, solver='lbfgs', maxiter=50, scoring='mse')
        aa_preds, conf_int = aarima_model.predict(n_periods = testx.shape[0], X = testx, return_conf_int = True)
        mse = mean_squared_error(aa_preds, testy)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(aa_preds, testy)
        mape = mean_absolute_percentage_error(aa_preds, testy)
        return aa_preds, mse, mae, mape, rmse

    def _linear_regressor(trainx, trainy, testx, testy):
        from sklearn.linear_model import LinearRegression 
        lr = LinearRegression()
        lr.fit(trainx, trainy)
        y_pred = lr.predict(testx)    
        mse = mean_squared_error(y_pred, testy)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_pred, testy)
        mape = mean_absolute_percentage_error(y_pred, testy)
        return y_pred, mse, mae, mape, rmse

    def _decision_tree_regressor(trainx, trainy, testx, testy):
        from sklearn.tree import DecisionTreeRegressor
        mse_list = []
        least_mse = 2
        for n in range(1, 50):
            model_dt = DecisionTreeRegressor(max_depth = n, splitter='best', criterion = 'mse', random_state = 42)
            model_dt.fit(trainx, trainy)
            y_pred = model_dt.predict(testx)
            error_score = model_dt.score(trainx, trainy)
            mse_list.append(error_score)
        mse_least = mse_list.index(max(mse_list))
        if mse_least == 0:
            mse_least = mse_least + 1
        model_dt = DecisionTreeRegressor(max_depth = mse_least, splitter='best', criterion = 'mse', random_state = 42)
        model_dt.fit(trainx, trainy)
        y_pred = model_dt.predict(testx)
        mse = mean_squared_error(y_pred, testy)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_pred, testy)
        mape = mean_absolute_percentage_error(y_pred, testy)
        return y_pred, mse, mae, mape, rmse

    class RandomForestRgrsr:
  
        def __init__(self, trainx = None, trainy = None, testx = None, testy = None):
            self.trainx = trainx
            self.trainy = trainy
            self.testx = testx
            self.testy = testy 

        def _rf_parameter_tuning(self):
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import ParameterGrid

            model_rf = RandomForestRegressor()
            grid = {'n_estimators' : [100, 150, 220, 250], 'max_depth': [3, 5, 8, 10], 'max_features': [1, 2], 'random_state': [42]}
            test_scores = []

            for g in ParameterGrid(grid):
                model_rf.set_params(**g)  
                model_rf.fit(self.trainx, self.trainy)
                test_scores.append(model_rf.score(self.testx, self.testy))

            best_idx = np.argmax(test_scores)
            #print(test_scores[best_idx], ParameterGrid(grid)[best_idx])
            return ParameterGrid(grid)[best_idx]

        def _rf_regressor(self, paramdict):
            from sklearn.ensemble import RandomForestRegressor
            model_rf = RandomForestRegressor(n_estimators = paramdict['n_estimators'], criterion = 'mse', max_depth = paramdict['max_depth'], max_features = paramdict['max_features'], bootstrap = True, random_state = 42)
            model_rf.fit(self.trainx, self.trainy)
            y_pred = model_rf.predict(self.testx)
            mse = mean_squared_error(y_pred, self.testy)
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(y_pred, self.testy)
            mape = mean_absolute_percentage_error(y_pred, self.testy)
            return y_pred, mse, mae, mape, rmse

    def _tfnn(trainx, trainy, testx, testy):
    
        """!pip install tensorflow
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense"""
        
        nn_model = Sequential()

        nn_model.add(Dense(8, activation = 'relu', input_dim = trainx.shape[1]))
        nn_model.add(Dense(16, activation = 'relu'))
        nn_model.add(Dense(8, activation = 'relu'))
        nn_model.add(Dense(1, activation = 'linear'))
        nn_model.compile(optimizer = 'Adam', loss = 'mse', metrics = ['mae'])

        print(nn_model.summary())

        history = nn_model.fit(trainx, trainy, epochs = 50)
        
        nn_eval = nn_model.evaluate(trainx, trainy)

        y_preds = nn_model.predict(testx)
    
        nn_mse = mean_squared_error(y_preds, testy)
        nn_rmse = np.sqrt(y_preds, testy)
        nn_mae = mean_absolute_error(y_preds, testy)
        nn_mape = mean_absolute_percentage_error(y_preds, testy)

        return y_preds, nn_mse, nn_mae, nn_mape, nn_rmse

    def _xgboost_reg_rscv(trainx, trainy, testx, testy):
    
        #xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        
        params = { 'max_depth': [3, 6, 10, 12, 15],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'n_estimators': [100, 250, 500, 750, 1000],
            'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.55, 0.7]}
        
        xgb_model = XGBRegressor(seed = 20)
        
        rscv_clf = RandomizedSearchCV(xgb_model, params, random_state=0, verbose = 0)

        rscv_clf.fit(trainx, trainy)
        
        best_params_df = pd.DataFrame([rscv_clf.best_params_])
        
        print(best_params_df)
        
        max_depth = best_params_df['max_depth'][0]
        lr = best_params_df['learning_rate'][0]
        n_est = best_params_df['n_estimators'][0]
        col_samples_bt = best_params_df['colsample_bytree'][0]
        
        xgb_best_model = XGBRegressor(colsample_bytree = col_samples_bt, learning_rate = lr, max_depth = max_depth, n_estimators = n_est)
        
        trained_model = xgb_best_model.fit(trainx, trainy) 
        
        xgb_preds = trained_model.predict(testx)
        
        xgb_mse = mean_squared_error(xgb_preds, testy)
        xgb_rmse = np.sqrt(xgb_mse)
        xgb_mae = mean_absolute_error(xgb_preds, testy)
        xgb_mape = mean_absolute_percentage_error(xgb_preds, testy)

        return xgb_preds, xgb_mse, xgb_mae, xgb_mape, xgb_rmse

    #Forecast Model Definition
    def _xgboost_reg(trainx, trainy, testx, testy, ):  #xgb_params
    
        xgb_model = XGBRegressor(colsample_bytree = 0.55, learning_rate = 0.01, max_depth = 6, n_estimators = 750, verbosity = 0)

        #xgb_model = XGBRegressor(colsample_bytree = xgb_params['colsample'], learning_rate = xgb_params['lr_rate'], max_depth = xgb_params['max_depth'], n_estimators = xgb_params['n_estimators'], verbosity = 0)
         
        trained_model = xgb_model.fit(trainx, trainy)
        
        xgb_preds = trained_model.predict(testx)
        
        xgb_mse = mean_squared_error(xgb_preds, testy)
        xgb_rmse = np.sqrt(xgb_mse)
        xgb_mae = mean_absolute_error(xgb_preds, testy)
        xgb_mape = mean_absolute_percentage_error(xgb_preds, testy)

        return xgb_preds, xgb_mse, xgb_mae, xgb_mape, xgb_rmse

    #Train Forecast by combination

    def _model_by_sequence(dframe, model_name, filter1, filter2, filter3, target):

        df_prepd = dframe.copy()
        result_bucket = pd.DataFrame()
        #params_df = pd.read_pickle('./source/xgb_params_table.pkl')

        for plant in df_prepd[filter1].unique():
            for dc in df_prepd[filter2].unique():
                for sku in df_prepd[filter3].unique():
                
                    query_result = pd.DataFrame(df_prepd.query("plant == @plant & distribution_channel == @dc & parent_name == @sku"))
                    query_result["Weekend_Indicator"] = query_result['Weekend_Indicator'].astype(int)
                    #query_result.drop(['plant_name', 'parent_code', 'parent_uom', 'converted_quantity'], axis = 1, inplace=True)
                    query_result.reset_index(inplace = True)
                    start_date = date_params['test_date_start'] #str(query_result['delivery_date'].max() - timedelta(22))
                    end_date = date_params['test_date_end'] #str(query_result['delivery_date'].max())
                    #query_result.drop(['level_0	', 'index'], axis = 1, inplace=True)
                    query_result.drop(['outliers', "seasonality", "day_of_month", "month", "Weekend_Indicator", "week_of_month", "y_ewm_14"], axis = 1, inplace=True)
                    query_result['y_ewm_7'] = pd.Series.ewm(query_result['y'], span=7, adjust=True).mean()
                    #query_result['y_ewm_14'] = pd.Series.ewm(query_result['y'], span=14, adjust=True).mean()
                    query_result['y_ewm_7_lag'] = query_result['y_ewm_7'].shift(query_result[query_result['delivery_date'] >= pd.to_datetime(start_date)].shape[0])
                    #query_result['cq_ewm_lag'].fillna(0)
                    #query_result['y_ewm_14_lag'] = query_result['y_ewm_14'].shift(query_result[query_result['delivery_date'] >= pd.to_datetime(start_date)].shape[0])
                    #query_result['cq_ewm_lag2'].fillna(0)
                    query_result['y_lag_qty'] = query_result[target].shift(query_result[query_result['delivery_date'] >= pd.to_datetime(start_date)].shape[0])
                    #query_result['y_7_pct_change_lag'] = query_result["y_7_pct_change"].shift(query_result[query_result['delivery_date'] >= pd.to_datetime(start_date)].shape[0])
                    #query_result['y_rol_lag_1_lag'] = query_result["y_rol_lag_1"].shift(query_result[query_result['delivery_date'] >= pd.to_datetime(start_date)].shape[0])
                    #query_result['y_lag_neg_3_lag'] = query_result["y_lag_neg_3"].shift(query_result[query_result['delivery_date'] >= pd.to_datetime(start_date)].shape[0])
                    query_result.replace(np.nan, 0, inplace = True)
                    print(query_result.info())
                    print(query_result.corr())
                    try:

                        print(query_result)

                        train_X, train_y, test_X, test_y, train_X_inv, test_X_inv = _split_train_test_bydate(query_result, 
                                                                                                            'delivery_date', 
                                                                                                            start_date, end_date, 
                                                                                                            target
                                                                                                            )
                        if model_name == 'linear reg':

                            lr_preds, mse, mae, mape, rmse = _linear_regressor(train_X, train_y, test_X, test_y)

                            lr_preds = pd.DataFrame(lr_preds, columns = ['Forecast'])
                            lr_preds.index = test_y.index
                            test_y = pd.DataFrame(test_y)
                            lrtest_df = pd.DataFrame(None)
                            lrtest_df = pd.concat([test_X_inv, test_y, np.round(lr_preds)], axis = 1)
                            lrtest_df

                            result_bucket = pd.concat([result_bucket, lrtest_df], axis = 0)

                        if model_name == 'auto arima':

                            aa_preds, mse, mae, mape, rmse = _model_arima(train_X, train_y, test_X, test_y)

                            aa_preds = pd.DataFrame(aa_preds, columns = ['Forecast'])
                            aa_preds.index = test_y.index
                            test_y = pd.DataFrame(test_y)
                            aatest_df = pd.DataFrame(None)
                            aatest_df = pd.concat([test_X_inv, test_y, np.round(aa_preds)], axis = 1)
                            aatest_df

                            result_bucket = pd.concat([result_bucket, aatest_df], axis = 0)

                        if model_name == 'decision tree':

                            dt_preds, mse, mae, mape, rmse = _decision_tree_regressor(train_X, train_y, test_X, test_y)

                            dt_preds = pd.DataFrame(dt_preds, columns = ['Forecast'])
                            dt_preds.index = test_y.index
                            test_y = pd.DataFrame(test_y)
                            dttest_df = pd.DataFrame(None)
                            dttest_df = pd.concat([test_X_inv, test_y, np.round(dt_preds)], axis = 1)
                            dttest_df

                            result_bucket = pd.concat([result_bucket, dttest_df], axis = 0)

                        if model_name == 'random forest reg':

                            rfr = RandomForestRgrsr(train_X, train_y, test_X, test_y)
                            bestparams = rfr._rf_parameter_tuning()
                            rfr_preds, rfr_mse, rfr_mae, rfr_mape, rfr_rmse = rfr._rf_regressor(bestparams)

                            rfr_preds = pd.DataFrame(rfr_preds, columns = ['Forecast'])
                            rfr_preds.index = test_y.index
                            test_y = pd.DataFrame(test_y)
                            rfrtest_df = pd.DataFrame(None)
                            rfrtest_df = pd.concat([test_X_inv, test_y, np.round(rfr_preds)], axis = 1)
                            rfrtest_df

                            result_bucket = pd.concat([result_bucket, rfrtest_df], axis = 0)

                        if model_name == 'xgboost':

                            xgb_preds, mse, mae, mape, rmse = _xgboost_reg(train_X, train_y, test_X, test_y)

                            xgb_preds = pd.DataFrame(xgb_preds, columns = ['Forecast'])
                            xgb_preds.index = test_y.index
                            test_y = pd.DataFrame(test_y)
                            xgbtest_df = pd.DataFrame(None)
                            xgbtest_df = pd.concat([test_X_inv, test_y, np.round(xgb_preds)], axis = 1)
                            xgbtest_df

                            result_bucket = pd.concat([result_bucket, xgbtest_df], axis = 0)

                        elif model_name == 'xgboost GSCV':
    
                            #xgbcv_preds, mse, mae, mape, rmse = _xgboost_reg_gscv(train_X, train_y, test_X, test_y)

                            xgbcv_preds = pd.DataFrame(xgbcv_preds, columns = ['Forecast'])
                            xgbcv_preds.index = test_y.index
                            test_y = pd.DataFrame(test_y)
                            xgbcvtest_df = pd.DataFrame(None)
                            xgbcvtest_df = pd.concat([test_X_inv, test_y, np.round(xgbcv_preds)], axis = 1)
                            xgbcvtest_df

                            result_bucket = pd.concat([result_bucket, xgbcvtest_df], axis = 0)

                        elif model_name == 'xgboost RSCV':
    
                            xgbcv_preds, mse, mae, mape, rmse = _xgboost_reg_rscv(train_X, train_y, test_X, test_y)

                            xgbcv_preds = pd.DataFrame(xgbcv_preds, columns = ['Forecast'])
                            xgbcv_preds.index = test_y.index
                            test_y = pd.DataFrame(test_y)
                            xgbcvtest_df = pd.DataFrame(None)
                            xgbcvtest_df = pd.concat([test_X_inv, test_y, np.round(xgbcv_preds)], axis = 1)
                            xgbcvtest_df

                            result_bucket = pd.concat([result_bucket, xgbcvtest_df], axis = 0)

                        elif model_name == 'gboost regressor':

                            #gbr_preds, mse, mae, mape, rmse = _gb_reg(train_X, train_y, test_X, test_y)

                            gbr_preds = pd.DataFrame(gbr_preds, columns = ['Forecast'])
                            gbr_preds.index = test_y.index
                            test_y = pd.DataFrame(test_y)
                            gbrtest_df = pd.DataFrame(None)
                            gbrtest_df = pd.concat([test_X_inv, test_y, np.round(gbr_preds)], axis = 1)
                            gbrtest_df

                            result_bucket = pd.concat([result_bucket, gbrtest_df], axis = 0)
                    except:
                        raise(Exception)
                        continue
                        
        return result_bucket, mse, mae, mape, rmse

    xgb_result, mse, mae, mape, rmse = _model_by_sequence(train_test_df, 'xgboost', 
                               'plant', 'distribution_channel', 'parent_name', 
                               'y'
                              )

    xgb_result.drop(['y', 'index'], axis = 1, inplace = True) #'day_of_week',

    df_plant_name = pd.DataFrame(df.groupby(['plant', 'plant_name'])['y'].count()).reset_index().drop('y', axis = 1)
    df_code_mat_uom = pd.DataFrame(df.groupby(['parent_code', 'parent_name', 'parent_uom'])['y'].count()).reset_index().drop('y', axis = 1)
    xgb_result.reset_index(inplace = True)
    xgb_merge = xgb_result.merge(df_plant_name, how = 'left', on = ['plant'])
    xgb_merge = xgb_merge.merge(df_code_mat_uom, how = 'left', on = ['parent_name'])
    xgb_merge = xgb_merge.rename(columns = {'plant': 'plant_code', 'parent_name': 'material_name', 'parent_code': 'material_number', 'Forecast': 'forecast_quantity', 'delivery_date':'date', 'parent_uom': 'uom'})
    xgbcv_forecast = xgb_merge[['plant_code', 'plant_name', 'material_number', 'material_name', 'uom', 'date', 'forecast_quantity', 'distribution_channel']]
    xgbcv_forecast['uid'] = xgbcv_forecast['plant_code'].astype(str) + '_' + xgbcv_forecast['material_number']
    xgbcv_forecast['uuid'] = xgbcv_forecast['uid'] + '_' + xgbcv_forecast['distribution_channel'].astype(str)
    xgbcv_forecast['forecast_quantity'] = abs(xgbcv_forecast['forecast_quantity'])
    xgbcv_forecast.to_pickle('./forecast_results/output_sumanth.pkl')
    xgbcv_forecast.to_csv('./forecast_results/xgb_results.csv')

    xgb_result_2, mse, mae, mape, rmse = _model_by_sequence(train_test_df, 'linear reg', 
                               'plant', 'distribution_channel', 'parent_name', 
                               'y'
                              )

    xgb_result_2.drop(['y', 'index'], axis = 1, inplace = True)

    df_plant_name = pd.DataFrame(df.groupby(['plant', 'plant_name'])['y'].count()).reset_index().drop('y', axis = 1)
    df_code_mat_uom = pd.DataFrame(df.groupby(['parent_code', 'parent_name', 'parent_uom'])['y'].count()).reset_index().drop('y', axis = 1)
    xgb_result_2.reset_index(inplace = True)
    xgb_merge_2 = xgb_result_2.merge(df_plant_name, how = 'left', on = ['plant'])
    xgb_merge_2 = xgb_merge_2.merge(df_code_mat_uom, how = 'left', on = ['parent_name'])
    xgb_merge_2 = xgb_merge_2.rename(columns = {'plant': 'plant_code', 'parent_name': 'material_name', 'parent_code': 'material_number', 'Forecast': 'forecast_quantity', 'delivery_date':'date', 'parent_uom': 'uom'})
    xgbcv_forecast_2 = xgb_merge_2[['plant_code', 'plant_name', 'material_number', 'material_name', 'uom', 'date', 'forecast_quantity', 'distribution_channel']]
    xgbcv_forecast_2['uid'] = xgbcv_forecast_2['plant_code'].astype(str) + '_' + xgbcv_forecast_2['material_number']
    xgbcv_forecast_2['uuid'] = xgbcv_forecast_2['uid'] + '_' + xgbcv_forecast_2['distribution_channel'].astype(str)
    xgbcv_forecast_2['forecast_quantity'] = abs(xgbcv_forecast_2['forecast_quantity'])
    xgbcv_forecast_2.to_pickle('./forecast_results/output_sumanth_2.pkl')
    xgbcv_forecast_2.to_csv('./forecast_results/xgb_results_2.csv')

    xgb_result_3, mse, mae, mape, rmse = _model_by_sequence(train_test_df, 'auto arima', 
                            'plant', 'distribution_channel', 'parent_name', 
                            'y'
                            )

    xgb_result_3.drop(['y', 'index'], axis = 1, inplace = True)

    df_plant_name = pd.DataFrame(df.groupby(['plant', 'plant_name'])['y'].count()).reset_index().drop('y', axis = 1)
    df_code_mat_uom = pd.DataFrame(df.groupby(['parent_code', 'parent_name', 'parent_uom'])['y'].count()).reset_index().drop('y', axis = 1)
    xgb_result_3.reset_index(inplace = True)
    xgb_merge_3 = xgb_result_3.merge(df_plant_name, how = 'left', on = ['plant'])
    xgb_merge_3 = xgb_merge_3.merge(df_code_mat_uom, how = 'left', on = ['parent_name'])
    xgb_merge_3 = xgb_merge_3.rename(columns = {'plant': 'plant_code', 'parent_name': 'material_name', 'parent_code': 'material_number', 'Forecast': 'forecast_quantity', 'delivery_date':'date', 'parent_uom': 'uom'})
    xgbcv_forecast_3 = xgb_merge_3[['plant_code', 'plant_name', 'material_number', 'material_name', 'uom', 'date', 'forecast_quantity', 'distribution_channel']]
    xgbcv_forecast_3['uid'] = xgbcv_forecast_3['plant_code'].astype(str) + '_' + xgbcv_forecast_3['material_number']
    xgbcv_forecast_3['uuid'] = xgbcv_forecast_3['uid'] + '_' + xgbcv_forecast_3['distribution_channel'].astype(str)
    xgbcv_forecast_3['forecast_quantity'] = abs(xgbcv_forecast_3['forecast_quantity'])
    xgbcv_forecast_3.to_pickle('./forecast_results/output_sumanth_3.pkl')
    xgbcv_forecast_3.to_csv('./forecast_results/xgb_results_3.csv')

    xgb_result_4, mse, mae, mape, rmse = _model_by_sequence(train_test_df, 'decision tree', 
                               'plant', 'distribution_channel', 'parent_name', 
                               'y')

    xgb_result_4.drop(['y', 'index'], axis = 1, inplace = True)

    df_plant_name = pd.DataFrame(df.groupby(['plant', 'plant_name'])['y'].count()).reset_index().drop('y', axis = 1)
    df_code_mat_uom = pd.DataFrame(df.groupby(['parent_code', 'parent_name', 'parent_uom'])['y'].count()).reset_index().drop('y', axis = 1)
    xgb_result_4.reset_index(inplace = True)
    xgb_merge_4 = xgb_result_4.merge(df_plant_name, how = 'left', on = ['plant'])
    xgb_merge_4 = xgb_merge_4.merge(df_code_mat_uom, how = 'left', on = ['parent_name'])
    xgb_merge_4 = xgb_merge_4.rename(columns = {'plant': 'plant_code', 'parent_name': 'material_name', 'parent_code': 'material_number', 'Forecast': 'forecast_quantity', 'delivery_date':'date', 'parent_uom': 'uom'})
    xgbcv_forecast_4 = xgb_merge_4[['plant_code', 'plant_name', 'material_number', 'material_name', 'uom', 'date', 'forecast_quantity', 'distribution_channel']]
    xgbcv_forecast_4['uid'] = xgbcv_forecast_4['plant_code'].astype(str) + '_' + xgbcv_forecast_4['material_number']
    xgbcv_forecast_4['uuid'] = xgbcv_forecast_4['uid'] + '_' + xgbcv_forecast_4['distribution_channel'].astype(str)
    xgbcv_forecast_4['forecast_quantity'] = abs(xgbcv_forecast_4['forecast_quantity'])
    xgbcv_forecast_4.to_pickle('./forecast_results/output_sumanth_4.pkl')
    xgbcv_forecast_4.to_csv('./forecast_results/xgb_results_4.csv')

    """xgb_result_5, mse, mae, mape, rmse = _model_by_sequence(train_test_df, 'random forest reg', 
                               'plant', 'distribution_channel', 'parent_name', 
                               'y_ewm_7'
                              )

    xgb_result_5.drop(['converted_quantity_ewm', 'index', 'day_of_week', 'day_of_month', 'week_of_month', 'Weekend_Indicator'], axis = 1, inplace = True)

    df_plant_name = pd.DataFrame(df.groupby(['plant', 'plant_name'])['converted_quantity'].count()).reset_index().drop('converted_quantity', axis = 1)
    df_code_mat_uom = pd.DataFrame(df.groupby(['parent_code', 'parent_name', 'parent_uom'])['converted_quantity'].count()).reset_index().drop('converted_quantity', axis = 1)
    xgb_result_5.reset_index(inplace = True)
    xgb_merge_5 = xgb_result_5.merge(df_plant_name, how = 'left', on = ['plant'])
    xgb_merge_5 = xgb_merge_5.merge(df_code_mat_uom, how = 'left', on = ['parent_name'])
    xgb_merge_5 = xgb_merge_5.rename(columns = {'plant': 'plant_code', 'parent_name': 'material_name', 'parent_code': 'material_number', 'Forecast': 'forecast_quantity', 'delivery_date':'date', 'parent_uom': 'uom'})
    xgbcv_forecast_5 = xgb_merge_5[['plant_code', 'plant_name', 'material_number', 'material_name', 'uom', 'date', 'forecast_quantity', 'distribution_channel']]
    xgbcv_forecast_5['uid'] = xgbcv_forecast_5['plant_code'].astype(str) + '_' + xgbcv_forecast_5['material_number']
    xgbcv_forecast_5['uuid'] = xgbcv_forecast_5['uid'] + '_' + xgbcv_forecast_5['distribution_channel'].astype(str)
    xgbcv_forecast_5['forecast_quantity'] = abs(xgbcv_forecast_5['forecast_quantity'])
    xgbcv_forecast_5.to_pickle('./forecast_results/output_sumanth_5.pkl')
    xgbcv_forecast_5.to_csv('./forecast_results/xgb_results_5.csv')"""

    xgb_result_6, mse, mae, mape, rmse = _model_by_sequence(train_test_df, 'xgboost RSCV', 
                               'plant', 'distribution_channel', 'parent_name', 
                               'y'
                              )

    xgb_result_6.drop(['y', 'index'], axis = 1, inplace = True)

    df_plant_name = pd.DataFrame(df.groupby(['plant', 'plant_name'])['y'].count()).reset_index().drop('y', axis = 1)
    df_code_mat_uom = pd.DataFrame(df.groupby(['parent_code', 'parent_name', 'parent_uom'])['y'].count()).reset_index().drop('y', axis = 1)
    xgb_result_6.reset_index(inplace = True)
    xgb_merge_6 = xgb_result_6.merge(df_plant_name, how = 'left', on = ['plant'])
    xgb_merge_6 = xgb_merge_6.merge(df_code_mat_uom, how = 'left', on = ['parent_name'])
    xgb_merge_6 = xgb_merge_6.rename(columns = {'plant': 'plant_code', 'parent_name': 'material_name', 'parent_code': 'material_number', 'Forecast': 'forecast_quantity', 'delivery_date':'date', 'parent_uom': 'uom'})
    xgbcv_forecast_6 = xgb_merge_6[['plant_code', 'plant_name', 'material_number', 'material_name', 'uom', 'date', 'forecast_quantity', 'distribution_channel']]
    xgbcv_forecast_6['uid'] = xgbcv_forecast_6['plant_code'].astype(str) + '_' + xgbcv_forecast_6['material_number']
    xgbcv_forecast_6['uuid'] = xgbcv_forecast_6['uid'] + '_' + xgbcv_forecast_6['distribution_channel'].astype(str)
    xgbcv_forecast_6['forecast_quantity'] = abs(xgbcv_forecast_6['forecast_quantity'])
    xgbcv_forecast_6.to_pickle('./forecast_results/output_sumanth_6.pkl')
    xgbcv_forecast_6.to_csv('./forecast_results/xgb_results_6.csv')

    return xgbcv_forecast, xgbcv_forecast_2, xgbcv_forecast_3, xgbcv_forecast_4, xgbcv_forecast_6  #xgb_result_5