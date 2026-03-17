# https://lanl-ansi.github.io/PowerModels.jl/stable/network-data/#The-Network-Data-Dictionary
# https://matpower.org/docs/ref/matpower5.0/caseformat.html
# https://github.com/gridfm/gridfm-datakit/blob/pm/scripts/notebooks/dispatch_with_powermodels.ipynb

from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.network import load_net_from_file
from gridfm_datakit.process.process_network import (
    init_julia,
    pf_preprocessing,
    pf_post_processing,
)
from gridfm_datakit.process.solvers import run_opf, run_pf
from gridfm_datakit.utils.column_names import (
    BUS_COLUMNS,
    GEN_COLUMNS,
    BRANCH_COLUMNS,
)

import os
import argparse
import math
import folium
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as mplt
from folium.plugins import HeatMap
from scipy.interpolate import griddata

# this is a function to generate df frames from the results - from Alban
def extract_results(processed_data, n_buses):
    bus_columns = BUS_COLUMNS
    bus_data = np.concatenate([item[0] for item in processed_data], axis=0)
    df_res_bus = pd.DataFrame(bus_data, columns=bus_columns)

    gen_data = np.concatenate([item[1] for item in processed_data], axis=0)
    df_res_gen = pd.DataFrame(
        gen_data,
        columns=GEN_COLUMNS,
    )

    branch_data = np.concatenate([item[2] for item in processed_data], axis=0)
    df_res_branch = pd.DataFrame(
        branch_data,
        columns=BRANCH_COLUMNS,
    )

    return df_res_bus, df_res_gen, df_res_branch


def get_columns(for_what):
    if for_what == 'bus':
        #return ['bus_i', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'area', 'Vm', 'Va', 'baseKV', 'zone', 'Vmax', 'Vmin']  PGLIB
        return ['bus_i','type','Pd','Qd','Gs','Bs','area','Vm','Va','baseKV','zone','Vmax','Vmin','lam_P','lam_Q','mu_Vmax','mu_Vmin']
    elif for_what == 'gen':
        # return ['bus_i','Pg','Qg','Qmax','Qmin','Vg','mBase','status','Pmax','Pmin'] PGLIB
        return ['bus_i','Pg','Qg','Qmax','Qmin','Vg','mBase','status','Pmax','Pmin','Pc1','Pc2','Qc1min','Qc1max','Qc2min','Qc2max','ramp_agc','ramp_10','ramp_30','ramp_q','apf','mu_Pmax','mu_Pmin','mu_Qmax','mu_Qmin']
    elif for_what == 'cost':
        return ['model', 'startup', 'shutdown', 'n', 'c2','c1','c0'] # not sure whether this is correct
    elif for_what == 'bran': 
        #return ['fbus','tbus','r','x','b','rateA','rateB','rateC','ratio','angle','status','angmin','angmax'] PGLIB
        return ['fbus','tbus','r','x','b','rateA','rateB','rateC','ratio','angle','status','angmin','angmax','Pf','Qf','Pt','Qt','mu_Sf','mu_St','mu_angmin','mu_angmax']
    else:
        print('ERROR')

#for_what has to be: bus,gen,cost,bran
def df_to_array(df,for_what,dtype=float):
    columns=get_columns(for_what)
    d = df
    arr = d.loc[:, columns].to_numpy(dtype=dtype, copy=True)
    return arr

def array_to_df(arr, for_what, copy=True):
    columns=get_columns(for_what)
    df = pd.DataFrame(arr, columns=columns, copy=copy)
    # Optional: make integer-like identifier columns actual integers
    int_like = ["bus_i", "type", "area", "zone", "fbus", "tbus", "status"]
    for c in int_like:
        if c in df.columns:
            df[c] = df[c].astype("int64")

    return df

def compare(opf_result,pf_result):
    processed_data = pf_post_processing(0, net, res=opf_result, res_dc=None, include_dc_res=False)
    processed_data = [(processed_data["bus"], processed_data["gen"], processed_data["branch"], processed_data["Y_bus"], processed_data["runtime"],)]
    df_res_bus_opf, df_res_gen_opf, df_res_bran_opf = extract_results(processed_data, n_buses)

    processed_data = pf_post_processing(0, net, res=pf_result, res_dc=None, include_dc_res=False)
    processed_data = [(processed_data["bus"], processed_data["gen"], processed_data["branch"], processed_data["Y_bus"], processed_data["runtime"],)]
    df_res_bus_pf, df_res_gen_pf, df_res_bran_pf = extract_results(processed_data, n_buses)

    mplt.figure(figsize=(12,20))
    fig, axs = mplt.subplots(3, 2)
    
    #x=net.res_bus.vm_pu; y=df_prior.vm_pu
    x=df_res_bus_pf.Vm; y=df_res_bus_opf.Vm
    axs[0, 0].set_title('Vm_pu'); axs[0, 0].plot([x.min(),x.max()],[x.min(),x.max()],color='black')
    axs[0, 0].scatter(x,y,color='orange'); axs[0, 0].set(xlabel='pf', ylabel='opf')

    x=df_res_bus_pf.Va; y=df_res_bus_opf.Va
    axs[0, 1].set_title('Va_degree'); axs[0, 1].scatter(x, y, color='blue')
    axs[0, 1].plot([x.min(),x.max()],[x.min(),x.max()],color='black'); axs[0, 1].set(xlabel='pf', ylabel='opf')

    x=df_res_bus_pf.Pd; y=df_res_bus_opf.Pd
    axs[1, 0].set_title('Pd_mw'); axs[1, 0].scatter(x, y, color='orange')
    axs[1, 0].plot([x.min(),x.max()],[x.min(),x.max()],color='black'); axs[1, 0].set(xlabel='pf', ylabel='opf')

    x=df_res_bus_pf.Qd; y=df_res_bus_opf.Qd
    axs[1, 1].set_title('Qd_mvar'); axs[1, 1].scatter(x, y, color='blue')
    axs[1, 1].plot([x.min(),x.max()],[x.min(),x.max()],color='black'); axs[1, 1].set(xlabel='pf', ylabel='opf')

    x=df_res_bus_pf.Pg; y=df_res_bus_opf.Pg
    axs[2, 0].set_title('Pg_mw'); axs[2, 0].scatter(x, y, color='orange')
    axs[2, 0].plot([x.min(),x.max()],[x.min(),x.max()],color='black'); axs[2, 0].set(xlabel='pf', ylabel='opf')
    
    x=df_res_bus_pf.Qg; y=df_res_bus_opf.Qg
    axs[2, 1].set_title('Qg_mvar'); axs[2, 1].scatter(x, y, color='blue')
    axs[2, 1].plot([x.min(),x.max()],[x.min(),x.max()],color='black'); axs[2, 1].set(xlabel='pf', ylabel='opf')
    
    fig.tight_layout()
    mplt.savefig('comparison.png')
    mplt.close()
    #mplt.show()

def result_postprocessing(results):
    # postprocessing
    hv_bus_violations=[]; lv_bus_violations=[]; old_violations =[]; increment=[]; bus=[]

    # if PF does not converged we set lv/hv violations to pf_max
    lv_max_pf=len(array_to_df(net.buses,'bus'))/2.0 ; hv_max_pf=len(array_to_df(net.buses,'bus'))/2.0
    # if OPF does not converge we set old_contingencies to numbers of lines / not trafos
    oc_max_opf=len(array_to_df(net.branches,'bran')[array_to_df(net.branches,'bran').ratio==0])
    # if OPF does not converge we set lv/hv violation to opf_max
    lv_max_opf=oc_max_opf*len(array_to_df(net.buses,'bus'))/2.0; hv_max_opf=oc_max_opf*len(array_to_df(net.buses,'bus'))/2.0

    for i in range(len(results['scenario'])):    
        hv_sum=0; lv_sum=0
        if results['opf'][i]==0:                                                                # in case OPF dispatch did not converged we set old violation = numbers of lines and hv/lv bus violations = number of buses/2 * numbers of lines
            old_violations.append(oc_max_opf); hv_sum=lv_max_opf; lv_sum=hv_max_opf   
            hv_bus_violations.append(hv_sum)
            lv_bus_violations.append(lv_sum)
        else:
            old_violations.append(len(results['line_contingency_violations'][i]))
            for j in range(len(results['line_contingency_violations'][i])):                    # this loops through each line and identifies whether it is an hv or lv violations 
                if results['line_contingency_violations'][i][j][1] == 'hv':
                    hv_sum=hv_sum+len(results['line_contingency_violations'][i][j][2])
                elif results['line_contingency_violations'][i][j][1] == 'lv':
                    lv_sum=lv_sum+len(results['line_contingency_violations'][i][j][2])
                elif results['line_contingency_violations'][i][0][1][0] == 'PF did not converged':
                    lv_sum=lv_max_pf; hv_sum=hv_max_pf
            hv_bus_violations.append(hv_sum)
            lv_bus_violations.append(lv_sum)
        # this is super inefficient - I fix it - see below
        # idx=results['scenario'][i]
        # pat = rf'^{idx}-'   # -> '^n-'   
        # increment.append(load.index[load.index.str.contains(pat)][0].split('-')[4])
    # bus.append(load.index[load.index.str.contains(pat)][0].split('-')[3])
        idx = results['scenario'][i]
        increment.append(load.index[idx*n_loads].split('-')[4])
        bus.append(load.index[idx*n_loads].split('-')[3])

    # create a df from result - not sure whether this is the best way of doing things
    df_results=pd.DataFrame(zip(results['scenario'],increment,bus,hv_bus_violations,lv_bus_violations,old_violations),columns=['scenario','increment','bus','number_of_hv_bus_violations','number_of_lv_bus_violations','old_violations'])
    df_results.set_index('scenario', inplace=True)
    df_results=df_results.astype({"increment": "float64","old_violations" :"float64","number_of_hv_bus_violations": "float64","number_of_lv_bus_violations": "float64", "old_violations": "float64"})

    #results['line_contingency_violations'][scenario][line][bus]
    return df_results

def power_balance(result, detail=False, baseMVA=100):

    processed_data = pf_post_processing(0, net, res=result, res_dc=None, include_dc_res=False)
    processed_data = [(processed_data["bus"], processed_data["gen"], processed_data["branch"], processed_data["Y_bus"], processed_data["runtime"],)]
    bus_df, gen_df, branch_df = extract_results(processed_data, n_buses)
    
    # in-service branches only (if present)
    br = branch_df.copy()
    if "br_status" in br.columns:
        br = br.loc[br["br_status"].fillna(1.0) == 1.0].copy()

    # totals from bus table (already MW/MVAr)
    bus_tot = bus_df.groupby("load_scenario_idx")[["Pg","Qg","Pd","Qd"]].sum()

    # shunts: convert PU -> MW/MVAr by multiplying by baseMVA
    if all(c in bus_df.columns for c in ["GS","BS","Vm"]):
        vm2 = bus_df["Vm"].astype(float) ** 2
        P_sh = (bus_df["GS"].astype(float) * vm2 * baseMVA)
        Q_sh = (-bus_df["BS"].astype(float) * vm2 * baseMVA)
        sh_tot = pd.DataFrame({
            "P_sh": P_sh.groupby(bus_df["load_scenario_idx"]).sum(),
            "Q_sh": Q_sh.groupby(bus_df["load_scenario_idx"]).sum(),
        })
    else:
        sh_tot = pd.DataFrame(index=bus_tot.index, data={"P_sh": 0.0, "Q_sh": 0.0})

    # branch “loss-like” sums (already MW/MVAr)
    br_tot = pd.DataFrame(index=bus_tot.index)
    br_tot["P_loss"]   = br.groupby("load_scenario_idx")["pf"].sum() + br.groupby("load_scenario_idx")["pt"].sum()
    br_tot["Q_series"] = br.groupby("load_scenario_idx")["qf"].sum() + br.groupby("load_scenario_idx")["qt"].sum()

    out = bus_tot.join(sh_tot, how="outer").join(br_tot, how="outer").fillna(0.0)

    out["P_balance_residual"] = out["Pg"] - out["Pd"] - out["P_sh"] - out["P_loss"]
    out["Q_balance_residual"] = out["Qg"] - out["Qd"] - out["Q_sh"] - out["Q_series"]

    if detail==True:
        print('P_load = '  ,out['Pd'].iloc[0])
        print('P_gen = '   ,out['Pg'].iloc[0])
        print('P_shunt = ' ,out['P_sh'].iloc[0])
        print('P_loss = '  ,out['P_loss'].iloc[0])
        print('***************************')
        print('Q_load = ' ,out['Qd'].iloc[0])
        print('Q_gen = '  ,out['Qg'].iloc[0])
        print('Q_shunt = ',out['Q_sh'].iloc[0])
        print('Q_loss = ',out['Q_series'].iloc[0])
   
    return float(out['P_balance_residual'].iloc[0]),float(out['Q_balance_residual'].iloc[0])  

def line_contingency(net,vmax=1.05, vmin=0.95, line_loading_max=1.0, pb=False):

    critical_lines = []
    pb_results=[]

    #turn array into df
    bran_df=array_to_df(net.branches,'bran')
    #for each line (exclude transformers for now)
    for i in bran_df[bran_df.ratio==0].index:
        #turn line off
        bran_df.loc[i,"status"]=0
        #convert df back to array
        net.branches=df_to_array(bran_df,'bran')
        #solve power flow
        #pf_result = run_pf(net, jl, fast=True)
        try:
            pf_result = run_pf(net, jl)
            #postprocessing of pf_result
            processed_data = pf_post_processing(0, net, res=pf_result, res_dc=None, include_dc_res=False)
            processed_data = [(processed_data["bus"], processed_data["gen"], processed_data["branch"], processed_data["Y_bus"], processed_data["runtime"],)]
            df_res_bus_pf, df_res_gen_pf, df_res_branch_pf = extract_results(processed_data, n_buses)    
            #check every bus for vmax validation
            if df_res_bus_pf.Vm.max() > vmax:
                critical_lines.append([i, 'hv', list(df_res_bus_pf.bus[df_res_bus_pf.Vm>vmax])])
            #check every bus for vmin validation
            if df_res_bus_pf.Vm.min() < vmin:
                critical_lines.append([i, 'lv', list(df_res_bus_pf.bus[df_res_bus_pf.Vm<vmin])])
            if pb==True:
                pb_results.append(power_balance(pf_result, detail=False))
                #print(power_balance(pf_result,detail=False)) 
        except:
            critical_lines.append([i,['PF did not converged']])
            if pb==True:
                pb_results.append('PF did not converged')
                
        #turn line back on
        bran_df.loc[i,"status"]=1
        net.branches=df_to_array(bran_df,'bran')

    if pb==True:
        return critical_lines,pb_results
    else:
        return critical_lines


# ********************************************************************
# **********************KEY INPUTS ***********************************
# ********************************************************************

case_name='case_ACTIVSg2000'
#case_name='case118_ieee'
#geofile='C:\\Users\\Coham\\OneDrive\\Desktop\\pyscripts\\projects\\Powermodels\\data\\case2000_goc_coordinates.csv'
#loadfile='C:\\Users\\Coham\\OneDrive\\Desktop\\pyscripts\\projects\\Powermodels\\data\\loads_case2000_goc.csv'
#resultfile='C:\\Users\\Coham\\OneDrive\\Desktop\\pyscripts\\projects\\Powermodels\\data\\case2000_results.pkl'

geofile='./case_ACTIVSg2000_coordinates.csv'
loadfile='./loads_case_ACTIVSg2000.csv'
resultfile=''
casefile='./case_ACTIVSg2000.m'

overwrite=True
max_iteration=2000
vmax=1.05
vmin=0.95

# Load network
net = load_net_from_file(casefile)
load = pd.read_csv(loadfile)
load.set_index('Unnamed: 0', inplace=True)

# Key parameters
n_buses = net.buses.shape[0]
id_buses =list(range(n_buses)) ; #all buses

id_loads=[]                    ; #all load buses
for i in range(n_buses):
    if net.buses[i][2] !=0:
        id_loads.append(i)
n_loads =len(id_loads)

id_sc=range(int(len(load)/n_loads))
n_sc=len(id_sc)

print(f"  Loaded {case_name}: {net.buses.shape[0]} buses, {net.gens.shape[0]} gens, {net.branches.shape[0]} branches, {n_loads} load buses, {int(len(load)/n_loads)} scenarios")

# ------------------------------------------------------------------
# Command-line arguments to split work across multiple jobs
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--start-index",
    type=int,
    required=True,
    help="Inclusive start index of scenario block (within [0, n_sc))",
)
parser.add_argument(
    "--end-index",
    type=int,
    required=True,
    help="Exclusive end index of scenario block (within [0, n_sc])",
)
parser.add_argument("--save-every", type=int, default=100,
                    help="Save a checkpoint parquet every N scenarios (default: 100)")
args = parser.parse_args()
start_index = args.start_index
end_index = args.end_index
save_every = args.save_every

output_dir = "./Texas2k-Prof-Output"
os.makedirs(output_dir, exist_ok=True)

def save_checkpoint(results, tag):
    """Save current results as parquet checkpoint."""
    if len(results['scenario']) == 0:
        return
    df = result_postprocessing(results)
    path = os.path.join(output_dir, f"df_results_{start_index}_{end_index-1}_{tag}.parquet")
    df.to_parquet(path)
    print(f"\n  [checkpoint] Saved {len(results['scenario'])} scenarios to {path}")

# create a visualizaton html 
bus_geodata = pd.read_csv(geofile,index_col=False)  # load geodata for visualization
# interactive_visualization(bus_geodata,case_name,net)

jl = init_julia(max_iter=max_iteration)

t0 = time.time()
if overwrite ==True:
    results= {'scenario':[],'opf':[],'line_contingency_violations':[]}
    scenarios_since_save = 0
    for idx in id_sc:
        # Only process scenarios in the requested sub-range
        if idx < start_index or idx >= end_index:
            continue
        #net = load_net_from_pglib(case_name)
        net = load_net_from_file(casefile)
        df_bus=array_to_df(net.buses,'bus')

        df_bus.loc[df_bus.Pd !=0,"Pd"]=list(load.p_mw.iloc[idx*n_loads:(idx+1)*n_loads])
        df_bus.loc[df_bus.Pd !=0,"Qd"]=list(load.q_mvar.iloc[idx*n_loads:(idx+1)*n_loads])

        # create array
        net.buses=df_to_array(df_bus,'bus')
        # run OPF
        try:
            opf_result = run_opf(net, jl)
            print(f" Scenario {idx}: {power_balance(opf_result,detail=False)}", end=" --- ")
            # set gen setpoints with OPF results
            net = pf_preprocessing(net, opf_result)
            results['scenario'].append(idx)
            results['opf'].append(1)
            results['line_contingency_violations'].append(line_contingency(net,vmax=vmax, vmin=vmin))
        except Exception as e:
            error_message = str(e)
            print(str(idx)+'  '+error_message, end=" --- ")
            results['scenario'].append(idx)
            results['opf'].append(0)
            results['line_contingency_violations'].append('NA')

        scenarios_since_save += 1
        if scenarios_since_save >= save_every:
            save_checkpoint(results, f"ckpt{len(results['scenario'])}")
            scenarios_since_save = 0

    # Save (skip if resultfile empty, e.g. when running chunked Slurm jobs)
    if resultfile:
        with open(resultfile, 'wb') as f:
            pickle.dump(results, f)
elif overwrite==False:
    # Load (requires resultfile to be set)
    if resultfile:
        with open(resultfile, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError("overwrite is False but resultfile is empty; cannot load results.")
t1 = time.time()
print('\nTime duration', t1-t0)
df_results=result_postprocessing(results)

# Save final results as parquet
parquet_path = os.path.join(output_dir, f"df_results_{start_index}_{end_index-1}.parquet")
df_results.to_parquet(parquet_path)
print(f"Saved df_results to {parquet_path}")
            
