import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ast 
from os import walk
import os
from tqdm import tqdm
import re

def plot_data(df, parameter, scale="log", errorbar=None, small=True, style=None):#("ci", 95)):
    if small:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        sns.lineplot(x=parameter, y='error', hue='method', style=style, data=df, errorbar=errorbar)
        plt.loglog()
        plt.legend(bbox_to_anchor=(-0.2, 1))
        plt.subplot(1,2,2)
        sns.lineplot(x=parameter, y='Accuracy_model', hue='method', style=style, data=df, errorbar=errorbar)
        plt.xscale(scale)
        plt.legend().remove()
    else:
        plt.figure(figsize=(20,5))
        plt.subplot(1,5,1)
        sns.lineplot(x=parameter, y='error', hue='method', style=style, data=df, errorbar=errorbar)
        plt.loglog()
        plt.legend(bbox_to_anchor=(-0.2, 1))
        plt.subplot(1,5,2)
        sns.lineplot(x=parameter, y='Accuracy_model', hue='method', style=style, data=df, errorbar=errorbar)
        plt.xscale(scale)
        plt.legend().remove()
        plt.subplot(1,5,3)
        sns.lineplot(x=parameter, y='Exact_model_found', hue='method', style=style, data=df, errorbar=errorbar)
        plt.xscale(scale)
        plt.legend().remove()
        plt.subplot(1,5,4)
        sns.lineplot(x=parameter, y='TP', hue='method', style=style, data=df, errorbar=errorbar)
        plt.xscale(scale)
        plt.legend().remove()
        plt.subplot(1,5,5)
        sns.lineplot(x=parameter, y='FP', hue='method', style=style, data=df, errorbar=errorbar)
        plt.xscale(scale)
        plt.legend().remove()
    
def add_column(df):
    l_TP, l_FP, l_FN, l_exact_model, l_accu, l_SBR_pareto_found = [], [], [], [], [], []
    for index, row in df.iterrows():
        l_TP.append(len(set(row["real_base"]).intersection(set(row["base_infered"]))) / len(set(row["real_base"]).union(row["base_infered"])))
        l_FP.append(len(set(row["base_infered"]).difference(set(row["real_base"]))) / len(set(row["real_base"]).union(row["base_infered"])))
        l_FN.append(len(set(row["real_base"]).difference(set(row["base_infered"]))) / len(set(row["real_base"]).union(row["base_infered"])))
        l_exact_model.append((set(row["base_infered"]) == set(row["real_base"]))*1)
        l_accu.append(len(set(row["real_base"]).intersection(set(row["base_infered"]))) / len(set(row["real_base"]).union(row["base_infered"])))
        l_SBR_pareto_found.append(bool(row["SBR_finds_real_model"]) and bool(row["real_model_on_pareto_front"]))
    df["TP"] = l_TP
    df["FP"] = l_FP
    df["FN"] = l_FN
    df["Exact_model_found"] = l_exact_model
    df["Accuracy_model"] = l_accu
    df["SBR_pareto_found"] = l_SBR_pareto_found
    
def aggreagate_csv_from_cluster(select_file = None, 
                                path=os.path.dirname(os.path.abspath('')) + "/csv",
                                ):
    print("Look in ", path)
    filesname = next(walk(path), (None, None, []))[2]  # [] if no file
    #filesname = ["name_truc.csv__1", "name_truc.csv__2"]
    d_name = {}
    print(filesname)
    for filename in filesname:
        l = filename.split("__")
        if len(l) == 2:
            d_name.setdefault(l[0], []).append(l[1])
    print(d_name.keys())
    for name in d_name.keys():
        if select_file is not None and select_file not in name:
            continue
        dataframes = []
        name_csv = path + "/" + name
        try: 
            os.remove(name_csv) 
        except:
            pass
        for i in tqdm(d_name[name], desc="Gather "+name):
            name_sub_file = name_csv+"__"+i
            # try:
            df = read_csv(name_sub_file)
            try:
                add_column(df)
            except Exception as e:
                print("Error: add_column : %s"%e)
            os.remove(name_sub_file)
            dataframes.append(df)

        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            #df.to_csv(name_csv, mode='a', header=not os.path.isfile(name_csv))
            # Save as a .pkl file
            if re.search(r'\.csv', name_csv):
                combined_df.to_csv(name_csv, header=not os.path.isfile(name_csv))
            elif re.search(r'\.pkl', name_csv):
                combined_df.to_pickle(name_csv) 

def read_csv(name_csv : str, converters={"base_infered":ast.literal_eval, "real_base" : ast.literal_eval, "init_params" : ast.literal_eval}):
    ## detect the type of name_csv : csv or pickle based on the .csv or .pkl
    if re.search(r'\.csv', name_csv):
        read = pd.read_csv 
        kwargs = {"converters" : converters}
    elif re.search(r'\.pkl', name_csv):
        read = pd.read_pickle
        kwargs = {}
    else:
        raise Exception("The file is not a csv or a pickle %s"%name_csv)
    try:
        df = read(name_csv, **kwargs)
    except Exception as e:
        print(e)
        try:
            name_csv_use = os.path.dirname(os.path.abspath('')) + "/csv/" + name_csv
            df = read(name_csv_use, **kwargs)
        except Exception as e:
            try:
                print(e)
                name_csv_use = os.path.abspath('') + "/csv/" + name_csv
                df = read(name_csv_use, **kwargs)
            except Exception as e:
                print("read fail %s"%e)
                df = pd.DataFrame()
    return df