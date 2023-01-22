def so_sku_preprocessing():

    import pandas as pd
    import numpy as np
    from pymongo import MongoClient
    from datetime import date, datetime, timedelta
    import json   

    dmsp_connection = 'mongodb://sales_orders_29nov:Rng3%26p%40ssw0rd@40.65.152.232:27017/sales_order_29nov?authSource=dms-picker&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false'
    vfdb_connection = 'mongodb://volumeforecasting:Volume%4021%21@40.65.152.232:27017/Volume_Forecasting?authSource=Volume_Forecasting&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false'
    benchmarker_con = 'mongodb://bmuser:bmuser123@40.65.152.232:27017/Volume_Forecasting?authSource=benchmarker'
    ppt_bm_con = "mongodb://bmuser:wx7Q%5EmqlE5%24z@52.163.118.48:27017/benchmarker?authSource=benchmarker"
    atlas_prod_bm = "mongodb+srv://bmuser:LwHjPgslHCA6Bdw6@benchmarkerprod.bvqga.mongodb.net/?retryWrites=true&w=majority"

    def _get_collection(connection_link, dbase, collection_name):

        # Connecting to MongoDB Client
        client = MongoClient(connection_link)
        # Fetching Volume Forecasting Database
        db = client[dbase]
        # List of all available collections
        db_collections = db.list_collection_names()
        # Returning the latest sales order collection from the DB
        return db, db[collection_name], db_collections

    plant_list = [1020]

    with open('./source/date_params.json', 'r') as f:
        date_json = json.load(f)

    date_params = date_json[0]

    try:
        dmsp_db, so_data, _ = _get_collection(atlas_prod_bm, 'benchmarker', 'sales_orders_mdb')
        new_query = {'req_del_date' :{'$gte': date_params['train_date_start'], '$lte': date_params['train_date_end']}, 'sales_document_type': {'$eq':'ZDOM'},
                    'sales_organization': {'$eq': 5000}, 'distribution_channel' : {'$in' : [50]}, #, 51, 52
                    'plant' : {'$in' : plant_list} #, 1009, 1017, 1020, 1028, 1049, 1056
                    }
        df_list = list(so_data.find(new_query), )
            
    except Exception as sodatareaderror:
        raise RuntimeError("sales_orders_29nov Data Ingestion failed")
    #print(df_list[0:10], "\n\n")
    #print(type(df_json[0:10]))
    df = pd.json_normalize(df_list, record_path='item', meta = ['req_del_date','created_at','sales_order_no', 'distribution_channel'])
    print(df.columns)
    df = df[df['rejection_reason'] !='W1']
    df['material_description'] = df['material_description'].str.lower()
    df = df.drop_duplicates(subset=['material_no','sales_order_no','order_quantity', 'material_description'],keep='last')
    df = df[['req_del_date', 'plant', 'material_no', 'material_description', 'uom', 'order_quantity', 'distribution_channel']]
    df = df.rename(columns={"req_del_date": "delivery_date"})
    
    df = df[df['plant'].isin(plant_list)]
    df = df[df['material_description'] == "coconut"]

    try: 
        ppt_bm_db, sku_col, _ = _get_collection(atlas_prod_bm, 'benchmarker', 'required_sku')
        sku_select = pd.DataFrame(list(sku_col.find()))
    except Exception as reqdskureaderror:
        raise RuntimeError("Error: required_sku Data Ingestion failed")

    sku_list = sku_select['material_code'].to_list()
    df = df[df['material_no'].isin(sku_list)]

    try:
        ppt_bm_db, pm_col, _ = _get_collection(atlas_prod_bm, 'benchmarker', 'child_parent_mapping')
        mmap = pd.DataFrame(list(pm_col.find()))
        conv_master = pd.DataFrame(list(pm_col.find()))
    except Exception as cpmreaderror:
        raise RuntimeError("child_parent_mapping Data Ingestion failed")

    mmap = mmap[['material_number', 'material_desc']]
    mmap.drop_duplicates(inplace = True)

    material_list = mmap['material_number'].to_list()
    df = df[df['material_no'].isin(material_list)]
    null_ind = df[df['material_description'].isnull()].index
    df.loc[null_ind, 'material_description'] = ''
    df['material_description'] = df['material_description'].apply(lambda x : 'remove' if 'scrapped' in x.lower().split() else x)
    df = df[df['material_description'] != 'remove']

    try:
        ppt_bm_con, pltname_col, _ = _get_collection(atlas_prod_bm, 'benchmarker', 'plant_with_name')
        plant_name_col = pd.DataFrame(list(pltname_col.find()))
    except Exception as pwnreaderror:
        raise RuntimeError("plant_with_name Data Ingestion failed")

    plant_name_col = plant_name_col[['plant_code', 'plant_name']]
    plant_name_col['plant_code'] = plant_name_col['plant_code'].astype(int)

    df = pd.merge(df, plant_name_col, how = 'left', left_on = 'plant', right_on = 'plant_code')
    df_merge = pd.merge(df, conv_master, how = 'left', left_on=['material_no','uom'], right_on = ['material_number','uom'])
    df_merge['net_weight'] = df_merge['net_weight'].astype(float)
    df_merge['order_quantity'].replace('', 0.0, inplace = True)
    df_merge['order_quantity'] = df_merge['order_quantity'].astype(float)
    df_merge['converted_quantity'] = df_merge['order_quantity'] * df_merge['net_weight']

    df_merge_grp = df_merge.groupby(['delivery_date','plant', 'plant_name', 'parent_code','parent_name','parent_uom', 'distribution_channel'])['converted_quantity'].sum().reset_index()

    remove = np.array(['WC0000000000200011','WC0000000000200034','WC0000000000200035','WC0000000000200024','WC0000000000260003','WC0000000000260016','WC0000000000250001','WC0000000000240002'])

    df_final = df_merge_grp[~df_merge_grp['parent_code'].isin(remove)]
    df = df_final.copy()

    df_unique = df.groupby(['plant', 'plant_name', 'parent_code', 'parent_name', 'parent_uom'])['distribution_channel'].count()

    df_unique = pd.DataFrame(df_unique).reset_index().drop('distribution_channel', axis = 1)

    df_plant_name = pd.DataFrame(df.groupby(['plant', 'plant_name'])['converted_quantity'].sum()).reset_index().drop('converted_quantity', axis = 1)

    #df_code_material = pd.DataFrame()
    df_code_material_uom = pd.DataFrame(df.groupby(['parent_code', 'parent_name', 'parent_uom'])['converted_quantity'].sum()).reset_index().drop('converted_quantity', axis = 1)

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
    
    def _add_date_metadata(dframe, datetime_column):
        dataframe = dframe.copy()
        dataframe[datetime_column] = pd.to_datetime(dataframe[datetime_column])
        #dataframe['quarter_of_year'] = dataframe[datetime_column].dt.quarter
        dataframe['month'] = dataframe[datetime_column].dt.month
        dataframe['day_of_month'] = dataframe[datetime_column].dt.day
        #dataframe['week_of_year'] = dataframe[datetime_column].dt.isocalendar().week
        dataframe['day_of_week'] = dataframe[datetime_column].dt.dayofweek
        dataframe['week_of_month'] = dataframe[datetime_column].apply(week_of_month)
        #dataframe['day_of_year'] = dataframe[datetime_column].dt.day_of_year
        #dataframe['days_in_month'] = dataframe[datetime_column].dt.daysinmonth
        #dataframe['season'] = dataframe[datetime_column].dt.month%12 // 3 + 1

        return dataframe

    def _categorize_data(dframe):
        dframe_copy = dframe.copy()
        dframe_copy['plant'] = dframe_copy['plant'].astype('category')
        dframe_copy['distribution_channel'] = dframe_copy['distribution_channel'].astype('category')
        dframe_copy['parent_name'] = dframe_copy['parent_name'].astype('category')
        #dframe_copy['quarter_of_year'] = dframe_copy['quarter_of_year'].astype('category')
        dframe_copy['month'] = dframe_copy['month'].astype('category')
        #dframe_copy['week_of_year'] = dframe_copy['week_of_year'].astype('category')
        dframe_copy['day_of_month'] = dframe_copy['day_of_month'].astype('category')
        dframe_copy['day_of_week'] = dframe_copy['day_of_week'].astype('category')
        #dframe_copy['day_of_year'] = dframe_copy['day_of_year'].astype('category')
        #dframe_copy['days_in_month'] = dframe_copy['days_in_month'].astype('category')
        #dframe_copy['season'] = dframe_copy['season'].astype('category')
        dframe_copy['converted_quantity'] = pd.to_numeric(dframe_copy['converted_quantity'])

        return dframe_copy

    def outlier_management(dframe, target):
    
        df_tce = dframe.copy()

        df_tce.set_index("delivery_date", inplace = True)
        
        from ThymeBoost import ThymeBoost as tb
        
        boosted_model = tb.ThymeBoost()
        
        output = boosted_model.detect_outliers(df_tce[target],
                                            trend_estimator='linear',
                                            seasonal_estimator='fourier',
                                            seasonal_period=7,
                                            global_cost='maicc',
                                            fit_type='global')

        df_tce.reset_index(inplace = True)

        print(output.head())
        
        df_tce_components_outliers = df_tce.merge(output.reset_index(), how = "left", on = ["delivery_date"]).drop(target, axis = 1)
        
        for n, i in enumerate(df_tce_components_outliers["outliers"]):
            if i == True:
                if df_tce_components_outliers["y"][n] > df_tce_components_outliers["yhat_upper"][n]:
                    df_tce_components_outliers["y"][n] = df_tce_components_outliers["yhat_upper"][n]
                if df_tce_components_outliers["y"][n] < df_tce_components_outliers["yhat_lower"][n]:
                    df_tce_components_outliers["y"][n] = df_tce_components_outliers["yhat"][n] - df_tce_components_outliers["y"][n]
            """if df_tce_components_outliers["y"][n] < 500:
                df_tce_components_outliers["y"][n] =  0.5 * df_tce_components_outliers["yhat_upper"][n]"""
        
        return df_tce_components_outliers

    df = outlier_management(df, "converted_quantity")

    df_sku = pd.DataFrame()

    for plant in list(df['plant'].unique()):
        for dc in list(df['distribution_channel'].unique()):
            for sku in list(df['parent_name'].unique()):
                df_cat_sku = df[(df['plant'] == plant) & (df['distribution_channel'] == dc) & (df['parent_name'] == sku)]
                df_cat_sku['delivery_date'] = pd.to_datetime(df_cat_sku['delivery_date'])
                df_cat_sku.set_index('delivery_date', inplace=True)
                billing_start_date = df_cat_sku.index.min()
                billing_recent_date = df_cat_sku.index.max()
                df_cat_sku.reset_index(inplace=True)
                try:
                    sku_date_range = pd.date_range(start=billing_start_date, end=billing_recent_date, freq='D')
                    df_cat_sku.set_index('delivery_date', inplace=True)
                    df_cat_sku = df_cat_sku.reindex(sku_date_range)
                    df_cat_sku['y'].interpolate(inplace = True)
                    #df_cat_sku['converted_quantity'] = df_cat_sku['converted_quantity'].fillna(method='ffill')
                    #df_cat_sku['y_ewm_7'] = pd.Series.ewm(df_cat_sku['y'], span = 7, adjust = True).mean()
                    #df_cat_sku['y_ewm_14'] = pd.Series.ewm(df_cat_sku['y'], span = 14, adjust = True).mean()
                    #df_cat_sku["y_lag"] = df_cat_sku["y"].shift(1)
                    #df_cat_sku["y_lag2"] = df_cat_sku["y"].shift(2)
                    #df_cat_sku["y_lag_neg_3"] = df_cat_sku["y"].rolling(window = 7, center=False, win_type="gaussian", method='single', axis = 0).mean(std = 0.5).shift(-3)
                    #df_cat_sku["y_7_pct_change"] = df_cat_sku["y_ewm_7"].pct_change(fill_method="ffill")
                    #df_cat_sku["y_rol_lag_1"] = df_cat_sku["y"].rolling(window = 7, center=False, win_type="gaussian", method='single', axis = 0).mean(std = 0.5).shift(1)
                    df_cat_sku.reset_index(inplace = True)
                    df_cat_sku['plant'] = plant; df_cat_sku['distribution_channel'] = dc; df_cat_sku['parent_name'] = sku
                    df_sku = pd.concat([df_sku, df_cat_sku])
                    
                except:
                    continue

    df_sku.rename(columns={"index" : "delivery_date"}, inplace=True)

    df_sku.drop(['plant_name', 'parent_code', 'parent_uom'], axis = 1, inplace = True)

    df_sku_merge = df_sku.merge(df_plant_name, how = 'left', on = ['plant'])
    df_sku_merge = df_sku_merge.merge(df_code_material_uom, how = 'left', on = ['parent_name'])
    df_sku_merge

    

    df_sku = df_sku_merge.copy()

    print(df_sku)

    df_meta = _add_date_metadata(df_sku, 'delivery_date')

    print(df_meta.info())
    
    #df_meta_cat = _categorize_data(df_meta)
    df_meta_cat = df_meta.copy()

    df_meta_cat.drop_duplicates(inplace = True)

    df_meta_cat.to_pickle('./source/SO_date_meta_cat_ewm.pkl')