import pandas as pd
import numpy as np
import time

def read_data(name):
    df = pd.read_csv(name)
    return df

def missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    Percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    
    x = pd.concat([total, Percentage], axis=1, keys=['Total','Percentage']) 
    return x  

#mean imputing
def mean_imputation(df):
    impute_loc = []
    start = time.time()
    for f in df.columns:
        index = []
        count = 0
        for j in range(len(df)):
            if df[f][j] == "?":
                df.loc[j,f] = 0
                count += 1
                index.append((f,j))
        df[f] = df[f].astype("float")
        avg = round((df[f].sum())/(len(df)-count),5)
        impute_loc.append(index)
        for idx  in index:
            df.loc[idx[1],idx[0]] = avg
    exe_time = time.time() - start
    return df,impute_loc ,exe_time

def manhatten_distance(ref,value):
    try:
        val = abs(float(ref)-float(value))
    except ValueError:
        val = 1
    return val

def HotDeckImputation(df):
    s = time.time()
    df_ = df.copy()
    for id in range(len(df)):
        arr = np.array(df.iloc[id,:])
        if "?" in arr :
            sliced_df = df.drop(id)
            impute_idx = np.where(arr == "?")[0]
            sub_df = pd.DataFrame({})
            for i in sliced_df.columns:
                ref = df.loc[id,i]
                sub_df[i+"_new"] = sliced_df[i].apply(lambda x:manhatten_distance(ref,x))
            sub_df['dist'] = sub_df.sum(axis=1)
            min_idx = sub_df['dist'].idxmin()
            #print(arr,id,min_idx)

            for k in impute_idx:
                if min_idx>=id:
                    min_idx = +1
                similar_obj = df.loc[min_idx,df.columns[k]]
                while similar_obj =="?":
                    min_idx = min_idx + 1
                    similar_obj = df.loc[min_idx,df.columns[k]]

                df_.loc[id,df.columns[k]] = similar_obj
    end = time.time()-s
    return df_,end 

def MeanAbsoluteError(df_orig,df,index):
    mae_sum = 0
    c=0
    for f in range(len(df_orig.columns)):
        for idx in index[f]:
            mae_sum += abs(df_orig.loc[idx[1],idx[0]] - df.loc[idx[1],idx[0]])
            c+=1
    mae = mae_sum/c
    return mae

def MAE_HD(orig_df,missing_df,imputed_df):
    mae_sum = 0
    c=0
    for id in range(len(orig_df)):
        arr = np.array(missing_df.iloc[id,:])
        if "?" in arr :
            impute_idx = np.where(arr == "?")[0]
            for i in impute_idx:
                c+=1
                mae_sum += abs(float(orig_df.loc[id,orig_df.columns[i]]) -float(imputed_df.loc[id,orig_df.columns[i]]))
    mae = mae_sum/c
    return mae


df_orig = read_data("dataset_complete.csv")
df_1perc = read_data("dataset_missing01.csv")
df_10perc = read_data("dataset_missing10.csv")

df1 = df_1perc.copy()
df2 = df_10perc.copy()


imputed_df1,impute_loc_1,t_1 = mean_imputation(df_1perc)
imputed_df2,impute_loc_2,t_2 = mean_imputation(df_10perc)

imputed_df1.to_csv('number_missing01_imputed_mean.csv',index=False)
imputed_df2.to_csv('number_ missing10_imputed_mean.csv',index = False)

print(f'MAE_01_mean {MeanAbsoluteError(df_orig,imputed_df1,impute_loc_1)}')
print(f'Runtime_01_mean {t_1*1000} mS')
print(f'MAE_02_mean {MeanAbsoluteError(df_orig,imputed_df2,impute_loc_2)}')
print(f'Runtime_02_mean {t_2*1000} mS')


imputed_hd_1,time_1 = HotDeckImputation(df1)
imputed_hd_2,time_2= HotDeckImputation(df2)

imputed_hd_1.to_csv('number_missing01_imputed_hd.csv',index=False)
imputed_hd_2.to_csv('number_ missing10_imputed_hd.csv',index = False)

print(f'MAE_01_hd : {MAE_HD(df_orig,df1,imputed_hd_1)}')
print(f'Runtime_01_hd: {time_1*1000} mS')
print(f'MAE_02_hd : {MAE_HD(df_orig,df2,imputed_hd_2)}')
print(f'Runtime_02_hd {time_2*1000} mS')




   
