import pandas as pd
import numpy as np
import json
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

    df.set_index('delivery_date', inplace = True)

    train_df = df.copy()

    today = str(date.today())[:10]

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
    "converted_quantity_ewm" : [],
    "converted_quantity_ewm2" : [],
    "converted_quantity" : []
    }
    for dates in list(pd.date_range(start = pd.to_datetime(today) + timedelta(2), end = pd.to_datetime(today) + timedelta(9), freq = 'D')):
        for i in train_df['plant'].unique():
            for j in train_df['distribution_channel'].unique():
                for k in train_df['parent_name'].unique():
                    
                    test_dict['delivery_date'].append(dates)
                    test_dict['plant'].append(i)
                    test_dict['distribution_channel'].append(j)
                    test_dict['parent_name'].append(k)
                    test_dict['converted_quantity_ewm'].append(0)
                    test_dict['converted_quantity_ewm2'].append(0)
                    test_dict['converted_quantity'].append(0)

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
    test_df['day_of_month'] = test_df['delivery_date'].dt.day
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
    train_test_df['Weekend_Indicator'] = train_test_df['Weekend_Indicator'].astype(int)

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
            train_x, train_y = X.loc[:pd.to_datetime(start_date) - timedelta(2)], y.loc[:pd.to_datetime(start_date) - timedelta(2)] 
            test_x, test_y = X.loc[(pd.to_datetime(start_date)):end_date], y.loc[(pd.to_datetime(start_date)):end_date]
            mle = MultiColumnLabelEncoder(columns = ['parent_name']) # 'day_of_week', 'Weekend_Indicator', 
            train_x = mle.fit_transform(train_x)
            train_x_inv = mle.inverse_transform(train_x)
            test_x = mle.fit_transform(test_x)
            test_x_inv = mle.inverse_transform(test_x)
        except: 
            dframe = dframe.reset_index()
            dframe = dframe.set_index(index_col)
            dframe = dframe.sort_index(ascending = True)
            X, y = dframe.loc[ : , dframe.columns != target_var ], dframe[target_var]
            train_x, train_y = X.loc[:pd.to_datetime(start_date) - timedelta(2)], y.loc[:pd.to_datetime(start_date) - timedelta(2)]
            test_x, test_y = X.loc[(pd.to_datetime(start_date)):end_date], y.loc[(pd.to_datetime(start_date)):end_date]
            mle = MultiColumnLabelEncoder(columns = ['parent_name']) #'day_of_week', 'Weekend_Indicator',
            train_x = mle.fit_transform(train_x)
            train_x_inv = mle.inverse_transform(train_x)
            test_x = mle.fit_transform(test_x)
            test_x_inv = mle.inverse_transform(test_x)
        return train_x, train_y, test_x, test_y, train_x_inv, test_x_inv

    def _xgboost_reg_rscv(trainx, trainy, testx, testy):
    
        #xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        
        params = { 'max_depth': [3,6,10],
            'learning_rate': [0.01, 0.03, 0.1],
            'n_estimators': [100, 250, 500, 750, 1000],
            'colsample_bytree': [0.1, 0.2, 0.3, 0.55]}
        
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
        
        query_result = df_prepd
        #query_result.drop(['plant_name', 'parent_code', 'parent_uom', 'converted_quantity', 'converted_quantity_ewm2', 'converted_quantity_ewm', 'converted_quantity_ewm4', 'converted_quantity_ewm5'], axis = 1, inplace=True)
        query_result.reset_index(inplace = True)
        start_date = str(test_df.index.min()) 
        end_date = str(test_df.index.max()) #str(query_result['delivery_date'].max())
        query_result['cq_ewm_lag'] = query_result['converted_quantity_ewm'].shift(query_result[query_result['delivery_date'] >= pd.to_datetime(start_date)].shape[0])
        query_result['cq_ewm_lag'].fillna(0)
        query_result['cq_ewm_lag2'] = query_result['converted_quantity_ewm2'].shift(query_result[query_result['delivery_date'] >= pd.to_datetime(start_date)].shape[0])
        query_result['cq_ewm_lag2'].fillna(0)
        query_result.drop(['converted_quantity_ewm', 'converted_quantity_ewm2'], axis = 1, inplace=True)

        if model_name == "xgboost":

            query_result['lag_qty'] = query_result[target].shift(query_result[query_result['delivery_date'] >= pd.to_datetime(start_date)].shape[0])
            query_result['lag_qty'].fillna(0, inplace = True)
            

        train_X, train_y, test_X, test_y, train_X_inv, test_X_inv = _split_train_test_bydate(query_result, 
                                                                                            'delivery_date', 
                                                                                            start_date, end_date, 
                                                                                            target
                                                                                            )

        if model_name == 'xgboost':

            xgb_preds, mse, mae, mape, rmse = _xgboost_reg(train_X, train_y, test_X, test_y)

            xgb_preds = pd.DataFrame(xgb_preds, columns = ['Forecast'])
            xgb_preds.index = test_y.index
            test_y = pd.DataFrame(test_y)
            xgbtest_df = pd.DataFrame(None)
            xgbtest_df = pd.concat([test_X_inv, test_y, np.round(xgb_preds)], axis = 1)
            xgbtest_df

            result_bucket = xgbtest_df    #pd.concat([result_bucket, xgbtest_df], axis = 0)

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

            result_bucket = xgbcvtest_df    #pd.concat([result_bucket, xgbcvtest_df], axis = 0)

        elif model_name == 'gboost regressor':

            #gbr_preds, mse, mae, mape, rmse = _gb_reg(train_X, train_y, test_X, test_y)

            gbr_preds = pd.DataFrame(gbr_preds, columns = ['Forecast'])
            gbr_preds.index = test_y.index
            test_y = pd.DataFrame(test_y)
            gbrtest_df = pd.DataFrame(None)
            gbrtest_df = pd.concat([test_X_inv, test_y, np.round(gbr_preds)], axis = 1)
            gbrtest_df

            result_bucket = pd.concat([result_bucket, gbrtest_df], axis = 0)
                    
        return result_bucket #, mse, mae, mape, rmse

    xgb_result = _model_by_sequence(train_test_df, 'xgboost', 
                               'plant', 'distribution_channel', 'parent_name', 
                               'converted_quantity'
                              )  #, mse, mae, mape, rmse

    print("\n Model 1 execution complete! \n")

    xgb_result_2 = _model_by_sequence(train_test_df, 'xgboost RSCV',
                                'plant', 'distribution_channel', 'parent_name', 
                                'converted_quantity'
                            )  #, mse, mae, mape, rmse

    print("\n Model 2 execution complete! \n")

    xgb_result.drop(['converted_quantity', 'index', 'day_of_week', 'day_of_month', 'season', 'quarter_of_year', 'month', 'week_of_month', 'Weekend_Indicator'], axis = 1, inplace = True)

    xgb_result_2.drop(['converted_quantity', 'index','day_of_week', 'day_of_month', 'season', 'quarter_of_year', 'month', 'week_of_month', 'Weekend_Indicator'], axis = 1, inplace = True)

    df_plant_name = pd.DataFrame(df.groupby(['plant', 'plant_name'])['converted_quantity'].count()).reset_index().drop('converted_quantity', axis = 1)
    df_code_mat_uom = pd.DataFrame(df.groupby(['parent_code', 'parent_name', 'parent_uom'])['converted_quantity'].count()).reset_index().drop('converted_quantity', axis = 1)
    
    xgb_result.reset_index(inplace = True)
    xgb_result_2.reset_index(inplace = True)

    reqd_date = str(date.today() + timedelta(2))[:10]

    xgb_merge = xgb_result.merge(df_plant_name, how = 'left', on = ['plant'])
    xgb_merge = xgb_merge.merge(df_code_mat_uom, how = 'left', on = ['parent_name'])
    xgb_merge = xgb_merge.rename(columns = {'plant': 'plant_code', 'parent_name': 'material_name', 'parent_code': 'material_number', 'Forecast': 'forecast_quantity', 'delivery_date':'date', 'parent_uom': 'uom'})
    xgbcv_forecast = xgb_merge[['plant_code', 'plant_name', 'material_number', 'material_name', 'uom', 'date', 'forecast_quantity', 'distribution_channel']]
    xgbcv_forecast['uid'] = xgbcv_forecast['plant_code'].astype(str) + '_' + xgbcv_forecast['material_number']
    xgbcv_forecast['uuid'] = xgbcv_forecast['uid'] + '_' + xgbcv_forecast['distribution_channel'].astype(str)
    xgbcv_forecast['forecast_quantity'] = abs(xgbcv_forecast['forecast_quantity'])
    xgbcv_forecast = xgbcv_forecast[xgbcv_forecast['date'] == reqd_date]
    xgbcv_forecast.to_pickle('./forecast_results/output_sumanth.pkl')
    xgbcv_forecast.to_csv('./forecast_results/xgb_results.csv')

    xgb_merge_2 = xgb_result_2.merge(df_plant_name, how = 'left', on = ['plant'])
    xgb_merge_2 = xgb_merge_2.merge(df_code_mat_uom, how = 'left', on = ['parent_name'])
    xgb_merge_2 = xgb_merge_2.rename(columns = {'plant': 'plant_code', 'parent_name': 'material_name', 'parent_code': 'material_number', 'Forecast': 'forecast_quantity', 'delivery_date':'date', 'parent_uom': 'uom'})
    xgbcv_forecast_2 = xgb_merge_2[['plant_code', 'plant_name', 'material_number', 'material_name', 'uom', 'date', 'forecast_quantity', 'distribution_channel']]
    xgbcv_forecast_2['uid'] = xgbcv_forecast_2['plant_code'].astype(str) + '_' + xgbcv_forecast_2['material_number']
    xgbcv_forecast_2['uuid'] = xgbcv_forecast_2['uid'] + '_' + xgbcv_forecast_2['distribution_channel'].astype(str)
    xgbcv_forecast_2['forecast_quantity'] = abs(xgbcv_forecast_2['forecast_quantity'])
    xgbcv_forecast_2 = xgbcv_forecast_2[xgbcv_forecast_2['date'] == reqd_date]
    xgbcv_forecast_2.to_pickle('./forecast_results/output_sumanth_2.pkl')
    xgbcv_forecast_2.to_csv('./forecast_results/xgb_results_2.csv')

    return xgbcv_forecast, xgbcv_forecast_2