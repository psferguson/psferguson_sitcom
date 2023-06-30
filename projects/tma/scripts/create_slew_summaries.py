import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

from astropy.time import Time, TimezoneInfo
from statsmodels.tsa.stattools import adfuller

from lsst.sitcom import vandv
from lsst.ts.idl.enums import MTM1M3

from lsst.summit.utils.tmaUtils import TMAEventMaker, TMAState
from lsst.summit.utils.efdUtils import getEfdData, makeEfdClient
from lsst_efd_client import EfdClient
from tqdm import tqdm

def get_slews(dayObs):
    # Select data from a given date
    eventMaker = TMAEventMaker()
    events = eventMaker.getEvents(dayObs)

    # Get lists of slew and track events
    slews = [e for e in events if e.type==TMAState.SLEWING]
    return slews

def get_univariate_splines(times, positions, velocities, interp_points, kernel_size=100, smoothing_factor=0.2):
        """
        """
        kernel=np.ones(kernel_size)/kernel_size
        try:
            posSpline = UnivariateSpline(times, position, s=0)
        except:
            #if there are duplicate time measurements remove them  (this occured on 
            # 23/11/22-23 and 21-22)
            times, indexes=np.unique(times, return_index=True)
            positions=positions[indexes]
            velocities=velocities[indexes]

            pos_spline = UnivariateSpline(times, positions, s=0)
            vel_spline1  = UnivariateSpline(times, velocities, s=0) 

        # Now smooth the derivative before differentiating again
        smoothed_vel = np.convolve(vel_spline1(interp_points), kernel, mode='same')
        vel_spline = UnivariateSpline(interp_points, smoothed_vel, s=smoothing_factor)
        acc_spline1 = vel_spline.derivative(n=1)
        smoothed_acc = np.convolve(acc_spline1(interp_points), kernel, mode='same')
        # Now smooth the derivative before differentiating again
        acc_spline = UnivariateSpline(interp_points, smoothed_acc, s=smoothing_factor)
        jerk_spline = acc_spline.derivative(n=1)
        return pos_spline(interp_points), vel_spline(interp_points), acc_spline(interp_points), jerk_spline(interp_points)


async def create_slew_summary_frame(slews,slew_config_frame):
    slew_summary_dict={"slew_num":[],
                   "begin_slew":[], 
                   "end_slew":[],
                   "begin_el":[],
                   "begin_az":[],
                   "delta_el":[],
                   "delta_az":[],
                   "max_torque_el":[],
                   "max_torque_az":[],
                   "abs_max_torque_el":[],
                   "abs_max_torque_az":[],
                   "min_torque_el":[],
                   "min_torque_az":[],
                   "max_hp_force":[],
                   "abs_max_hp_force":[],
                   "min_hp_force":[],
                   "max_rstd_hp_force":[],
                   "max_vel_az":[],
                   "max_acc_az":[],
                   "max_jerk_az":[],
                   "max_vel_el":[],
                   "max_acc_el":[],
                   "max_jerk_el":[],
                
                       
                   }
    client = makeEfdClient()
    for i_slew in tqdm(range(len(slews))):
        df_mtmount_el = getEfdData(client,'lsst.sal.MTMount.elevation', event=slews[i_slew], 
                                    postPadding=1, prePadding=1)
        if  df_mtmount_el.shape == (0,0):
            continue
        df_mtmount_az = getEfdData(client,'lsst.sal.MTMount.azimuth', event=slews[i_slew], 
                                    postPadding=1, prePadding=1)
        if  df_mtmount_az.shape == (0,0):
            continue

        df_m1m3_hardpoint = getEfdData(client,"lsst.sal.MTM1M3.hardpointActuatorData", event=slews[i_slew], 
                                    postPadding=1, prePadding=1)
        if  df_m1m3_hardpoint.shape == (0,0):
            continue
        slew_summary_dict["slew_num"].append(i_slew)
        slew_summary_dict["begin_slew"].append(slews[i_slew].begin.utc.unix)
        slew_summary_dict["end_slew"].append(slews[i_slew].end.utc.unix)
        slew_summary_dict["begin_el"].append(df_mtmount_el["actualPosition"][0])
        slew_summary_dict["begin_az"].append(df_mtmount_az["actualPosition"][0])
        slew_summary_dict["delta_el"].append(df_mtmount_el["actualPosition"][0] - df_mtmount_el["actualPosition"][-1])
        slew_summary_dict["delta_az"].append(df_mtmount_az["actualPosition"][0] - df_mtmount_az["actualPosition"][-1])
        slew_summary_dict["max_torque_el"].append((df_mtmount_el["actualTorque"].max()))
        slew_summary_dict["max_torque_az"].append((df_mtmount_az["actualTorque"].max()))
        slew_summary_dict["abs_max_torque_el"].append(abs(df_mtmount_el["actualTorque"].max()))
        slew_summary_dict["abs_max_torque_az"].append(abs(df_mtmount_az["actualTorque"].max()))
        slew_summary_dict["min_torque_el"].append(df_mtmount_el["actualTorque"].min())
        slew_summary_dict["min_torque_az"].append(df_mtmount_az["actualTorque"].min())

        maxima_hp=[]
        maxima_rstd_hp=[]
        for i in range(6):
            maxima_hp.append(((df_m1m3_hardpoint[f"measuredForce{i}"]).max(), i))
            maxima_hp.append((df_m1m3_hardpoint[f"measuredForce{i}"].min(), i))
            maxima_rstd_hp.append(df_m1m3_hardpoint[f"measuredForce{i}"].rolling(100).std())
        maxima_hp=np.array(maxima_hp)

        slew_summary_dict["abs_max_hp_force"].append(abs(maxima_hp.max()))
        slew_summary_dict["max_hp_force"].append(maxima_hp.max())
        slew_summary_dict["max_rstd_hp_force"].append(maxima_hp.max()) 
        slew_summary_dict["min_hp_force"].append(maxima_hp.min())
        
        # slew profiles 
        az_ps = df_mtmount_az['actualPosition'].values
        az_vs = df_mtmount_az['actualVelocity'].values
        az_xs = df_mtmount_az['timestamp'].values - df_mtmount_az['timestamp'].values[0]  
        
        el_ps = df_mtmount_el['actualPosition'].values
        el_vs = df_mtmount_el['actualVelocity'].values
        el_xs = df_mtmount_el['timestamp'].values - df_mtmount_el['timestamp'].values[0]
        
        npoints=int(np.max([np.round((az_xs[-1]-az_xs[0])/0.01/1e3,0)*1e3, 4000]))
        plot_az_xs = np.linspace(az_xs[0], az_xs[-1], npoints)
        plot_el_xs = np.linspace(el_xs[0], el_xs[-1], npoints)
        
        az_pos_spline, az_vel_spline, az_acc_spline, az_jerk_spline = get_univariate_splines(az_xs, az_ps, az_vs, plot_az_xs) 
        el_pos_spline, el_vel_spline, el_acc_spline, el_jerk_spline = get_univariate_splines(el_xs, el_ps, el_vs, plot_el_xs)
        
        slew_summary_dict["max_vel_az"].append( np.max(az_vel_spline))
        slew_summary_dict["max_acc_az"].append(np.max(az_acc_spline))
        slew_summary_dict["max_jerk_az"].append( np.max(az_jerk_spline))
        slew_summary_dict["max_vel_el"].append( np.max(el_vel_spline))
        slew_summary_dict["max_acc_el"].append(np.max(el_acc_spline))
        slew_summary_dict["max_jerk_el"].append( np.max(el_jerk_spline))
        

    slew_summary_frame=pd.DataFrame(slew_summary_dict)
    # force balanacing 
    fa_state=getEfdData(client,"lsst.sal.MTM1M3.logevent_forceActuatorState", begin=slews[0].begin, end=slews[-1].end, prePadding=600)
    fa_state["snd_timestamp_utc"]=Time(fa_state["private_sndStamp"], format="unix_tai").unix
    fa_state=fa_state.loc[:,["snd_timestamp_utc","balanceForcesApplied"]]
    
    balance_state=[]
    #hardpoint_dict={"left_only":"enabled","right_only":"disabled"}
    for i in slew_summary_frame.index:
        val=slew_summary_frame.loc[i,"begin_slew"]
        sel=(fa_state["snd_timestamp_utc"] < val)
        if sel.sum() > 0:
            state=fa_state.iloc[np.argmax(fa_state["snd_timestamp_utc"][sel]),1]
            balance_state.append(state)
        else: 
            balance_state.append(np.nan)
    
    slew_summary_frame["balanceForcesApplied"]=balance_state
    
    # slew configs
    motion_state=[]
    slew_config_times=np.array([Time(i) for i in slew_configs_frame["time"].values])
    for i in slew_summary_frame.index:
        val=slew_summary_frame.loc[i,"begin_slew"]
        sel=(slew_config_times < Time(val, format="unix").datetime)
        
        if sel.sum() > 0:
            
            state=slew_config_frame.iloc[np.argmax(slew_config_frame["time"][sel]),1]
        else:
            state=np.nan
        motion_state.append(state)
    slew_summary_frame["motion_state"]=motion_state
    return slew_summary_frame
if __name__ == '__main__':
    client = makeEfdClient()
    for dayObs in [20230627,20230628]:
        print(dayObs)
        slews = get_slews(dayObs)
        slew_configs_frame = pd.read_csv(f"./data/slew_configs/motion_settings_{dayObs}.csv")
       
        slew_summary_frame = asyncio.run(create_slew_summary_frame(slews, slew_configs_frame))
        slew_summary_frame.to_csv(f"./data/slew_summaries/slew_summary_frame_{dayObs}.csv", index=False)
    