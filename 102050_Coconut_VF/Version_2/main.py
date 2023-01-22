from pymongo import MongoClient
import pandas as pd
import time
from datetime import date
from preprocessing import data_preprocessing
from forecast_with_params import forecast_volume

def main():

    print("\n Date of Execution: {}".format(date.today()))

    start_time = time.time()

    print("\n Execution Start Time: {}".format(start_time))

    print("\n ***************Preprocessing Data****************** \n")

    data_preprocessing()

    print(" Data Preprocessed Successfully \n")

    print(" *****************Forecasting Volume******************** \n")

    result_df, result_df_2, result_df_3, result_df_4, result_df_6 = forecast_volume()  #result_df_5,      

    print("Forecast Model Execution Successful \n")

    end_time = time.time()

    print("\n Execution End Time: {} \n".format(end_time))

    runtime = (end_time - start_time) / 60

    print("\n Total Runtime in minutes: {} \n".format(runtime))

    result_df['run_date'] = pd.to_datetime(str(date.today())[:10])

    result_df_2['run_date'] = pd.to_datetime(str(date.today())[:10])

    result_df_3['run_date'] = pd.to_datetime(str(date.today())[:10])

    result_df_4['run_date'] = pd.to_datetime(str(date.today())[:10])

    #result_df_5['run_date'] = pd.to_datetime(str(date.today())[:10])

    result_df_6['run_date'] = pd.to_datetime(str(date.today())[:10])

    result_df['script_runtime'] = runtime

    result_df_2['script_runtime'] = runtime

    result_df_3['script_runtime'] = runtime

    result_df_4['script_runtime'] = runtime

    #result_df_5['script_runtime'] = runtime

    result_df_6['script_runtime'] = runtime

    client = MongoClient('mongodb+srv://uatuser:WnQ9wz4ccWLpjqXB@cluster0.hotvf.mongodb.net/benchmarkeruat')

    db = client['benchmarkeruat']

    result_df.to_pickle('./forecast_results/output_sumanth_final.pkl')
    result_df.to_csv('./forecast_results/xgboost_result.csv')
    
    result_df_2.to_pickle('./forecast_results/output_sumanth_final_2.pkl')
    result_df_2.to_csv('./forecast_results/lr_result.csv')

    result_df_3.to_pickle('./forecast_results/output_sumanth_final_3.pkl')
    result_df_3.to_csv('./forecast_results/aarima_result.csv')

    result_df_4.to_pickle('./forecast_results/output_sumanth_final_4.pkl')
    result_df_4.to_csv('./forecast_results/dt_result.csv')

    #result_df_5.to_pickle('./forecast_results/output_sumanth_final_5.pkl')
    #result_df_5.to_csv('./forecast_results/output_sumanth_final_5.csv')

    result_df_6.to_pickle('./forecast_results/output_sumanth_final_6.pkl')
    result_df_6.to_csv('./forecast_results/xgboost_rscv_result.csv')

    print("\n Pushing Forecast results to the DB \n")

    #db.output_sumanth.insert_many(result_df_2.to_dict('records'))

    print("\n Results successfully inserted into collection A")

    #db.output_sumanth_2.insert_many(result_df.to_dict('records'))

    print("\n Results successfully insterted into collection B")

if __name__ == "__main__":
    main()