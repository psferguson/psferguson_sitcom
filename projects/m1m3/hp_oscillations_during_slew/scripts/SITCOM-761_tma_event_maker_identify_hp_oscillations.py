"""
Currently this script tries to identify oscillation events in the 
'lsst.sal.MTM1M3.hardpointActuatorData' measuredForces

For now set the start_date, end_date and window (seconds) and the script 
will break up EFD queries search for events and save the results in a `./data` 
directory. 
"""

import argparse
import asyncio
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.time import Time, TimeDelta
from scipy.interpolate import UnivariateSpline

from lsst_efd_client import EfdClient
from lsst.summit.utils.tmaUtils import TMAEventMaker, TMAState
from lsst.summit.utils.efdUtils import getEfdData, makeEfdClient, calcNextDay


from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from tqdm import tqdm


class identify_oscillation_events():
    def __init__(self):
        self.force="two"
        self.rolling_std_window = 100  # 100 is ~ 2 second window
        self.association_window_1 = 2 # window in seconds to combine peaks in same actuator
        self.association_window_2 = 4 # window in seconds to combine peaks accross actuators
        self.slew_speed_min = 0.01 # used for identifiying when we are slewing
        self.peak_height=100
        # self.output_dir=output_dir
        # self.smoothingFactor=0.2 # In spline creation
        # self.kernel_size=100 # In convolution
        # self.buffer = 4
        # self.return_meas=True
        # self.day=self.start_time.iso[:10]
        # self.fit_method=fit_method # either splines or savgol
        # self.window = 400 # savgol window size
    def add_timestamp(self,data):
        # add correct timestamp column in utc
        if "snd_timestamp_utc" not in data.columns:
            data["snd_timestamp_utc"]=Time(data["private_sndStamp"], format = "unix_tai").unix
        return data
    def combine_peaks_accross_actuators(self, peak_frame, window=4):
        # takes a set of identified peaks in different actuators and associates nearby 
        # ones default window is 4 seconds, 
        # returns 1 peak per window and the max height of peaks in that window as well as the number of 
        # actuators that saw that peak, only returns peaks seen with more than 3 actuators.
        super_heights =[]
        super_times = []
        super_counts = [] 
        super_actuators= []
        super_rmean =[]
        for peak in peak_frame["times"].values:
            sel = (abs(peak_frame["times"]-peak) < window)
            subframe=peak_frame[sel]
            count=len(np.unique(subframe["actuators"]))
            if count > 3:
                max_height=subframe["heights"].max()
                super_counts.append(count)
                super_heights.append(max_height)
                super_times.append(subframe["times"][subframe["heights"]==max_height].values)
                super_actuators.append(subframe["actuators"][subframe["heights"]==max_height].values)
                super_rmean.append(subframe["rmean_diff"][subframe["heights"]==max_height].values)
        if len(super_times)==0:
            return pd.DataFrame({"times":super_times,
                             "heights":super_heights,
                             "rmean":super_rmean,
                             "counts":super_counts, 
                             "actuators":super_actuators})
        super_times = np.concatenate(super_times)
        super_actuators = np.concatenate(super_actuators)
        super_rmean=np.concatenate(super_rmean)
        super_heights=np.array(super_heights)
        super_counts=np.array(super_counts)
        
        
        super_times, super_inds=np.unique(super_times, return_index=True)
        super_heights=super_heights[super_inds]
        super_actuators=super_actuators[super_inds]
        super_counts=super_counts[super_inds]
        super_rmean=super_rmean[super_inds]
        #import pdb;pdb.set_trace()
        return pd.DataFrame({"times":super_times,
                             "heights":super_heights,
                             "rmean":super_rmean,
                             "counts":super_counts, 
                             "actuators":super_actuators})
    async def get_slews(self,day_obs):
        eventMaker = TMAEventMaker()
        events = eventMaker.getEvents(int(day_obs))
        slews = [e for e in events if e.type==TMAState.SLEWING]
        return slews
    
    async def get_data(self, event, client):
            "Extract all the MTMount data from the EFD and save to parquet files"

            # Get EFD client options are usdf_efd or summit_efd
            
            self.query_dict={}
            self.query_dict["day_obs"]=event.dayObs
            self.query_dict["seq_num"]=event.seqNum
            self.query_dict["el"] = getEfdData(client,
                                               'lsst.sal.MTMount.elevation',
                                               event=event, 
                                               prePadding=5,
                                               postPadding=5
                                              )
            
            if  ("private_sndStamp" not in self.query_dict["el"].keys()):
                print("no el data")
                self.query_dict=None
                return
            self.query_dict["el"]=self.add_timestamp(self.query_dict["el"])
            
            
            self.query_dict["az"] = getEfdData(client,
                                               'lsst.sal.MTMount.azimuth',
                                               event=event, 
                                               prePadding=5,
                                               postPadding=5
                                              )
            
            if  ("private_sndStamp" not in self.query_dict["az"].keys()):
                print("no az data")
                self.query_dict=None
                return
            self.query_dict["az"]=self.add_timestamp(self.query_dict["az"])
            

            self.query_dict["hpmf"] = getEfdData(client,
                                                 'lsst.sal.MTM1M3.hardpointActuatorData',
                                                 event=event, 
                                                 prePadding=5,
                                                 postPadding=5,
                                                 columns=['private_sndStamp',
                                                          'measuredForce0', 
                                                          'measuredForce1', 
                                                          'measuredForce2', 
                                                          'measuredForce3',
                                                          'measuredForce4', 
                                                          'measuredForce5'
                                                         ],  
                                                )
            if  ("private_sndStamp" not in self.query_dict["hpmf"].keys()):
                print("no hpmf data")
                self.query_dict=None
                return
            self.query_dict["hpmf"] = self.add_timestamp(self.query_dict["hpmf"])
    def identify(self, query_dict):
        if query_dict is None:
            return None
        
        peak_dict={}
        peak_frame=pd.DataFrame({"times":[],"heights":[],"actuators":[]})
        for i in range(6):
            # this loop identifies rolling std peaks in the measured force
            rolling_std_val=self.query_dict["hpmf"][f"measuredForce{i}"].rolling(self.rolling_std_window).std() # 100 is ~ 2 second window
            rolling_mean_val=self.query_dict["hpmf"][f"measuredForce{i}"].rolling(self.rolling_std_window).mean()
            peak_indicies=find_peaks(rolling_std_val, height=self.peak_height )[0] 
            
                
            # keep time and height of peaks
            peak_dict[f"hp_{i}_peak_times"]=self.query_dict["hpmf"]["snd_timestamp_utc"][peak_indicies].values
            peak_dict[f"hp_{i}_peak_heights"]= rolling_std_val[peak_indicies].values
            start_ind = [np.max([i - 500,0]) for i in peak_indicies]
            stop_ind = [np.min([i + 500, len(rolling_mean_val)-1]) for i in peak_indicies]
            peak_dict[f"hp_{i}_peak_rmean_diff"] = rolling_mean_val[stop_ind].values - rolling_mean_val[start_ind].values
            
            # for each peak combine by looking at all peaks within 
            # a window and keeping the one with the largest height then np.unique that 
            super_heights=[]
            super_times=[]
            super_rmean=[]
            
            for j,peak in enumerate(peak_dict[f"hp_{i}_peak_times"]):
                sel_peaks=(abs(peak_dict[f"hp_{i}_peak_times"]-peak) < self.association_window_1)
                max_height=np.max(peak_dict[f"hp_{i}_peak_heights"][sel_peaks])
                #max_rmean=np.max(peak_dict[f"hp_{i}_peak_long_mean"][sel_peaks])
                max_time=peak_dict[f"hp_{i}_peak_times"][sel_peaks][np.where(peak_dict[f"hp_{i}_peak_heights"][sel_peaks]==max_height)]
                max_index=np.where(peak_dict[f"hp_{i}_peak_times"]==max_time)[0]
                super_times.append(peak_dict[f"hp_{i}_peak_times"][max_index])
                super_heights.append(peak_dict[f"hp_{i}_peak_heights"][max_index])
                super_rmean.append(peak_dict[f"hp_{i}_peak_rmean_diff"][max_index])
            peak_dict[f"hp_{i}_peak_times"] = np.unique(super_times)
            peak_dict[f"hp_{i}_peak_heights"] = np.unique(super_heights)
            peak_dict[f"hp_{i}_peak_rmean_diff"] = np.unique(super_rmean)
            
            peak_frame=pd.concat([peak_frame,pd.DataFrame({"times":peak_dict[f"hp_{i}_peak_times"],
                                                           "heights":peak_dict[f"hp_{i}_peak_heights"],
                                                           "rmean_diff": peak_dict[f"hp_{i}_peak_rmean_diff"],
                                                           "actuators":i})])
            peak_frame=peak_frame.sort_values("times")

            # next we want to combine peaks across actuators
            overall_frame=self.combine_peaks_accross_actuators(peak_frame, window=self.association_window_2)

            #identify when we are slewing
            overall_frame["slew_state"]=False
            slew_speed_el=interp1d(self.query_dict["el"]["snd_timestamp_utc"],
                                abs(self.query_dict["el"]["actualVelocity"].rolling(10).mean()),
                                bounds_error=False)
            slew_speed_az=interp1d(self.query_dict["az"]["snd_timestamp_utc"],
                                abs(self.query_dict["az"]["actualVelocity"].rolling(10).mean()),
                                bounds_error=False)

            slew_velocity_el=interp1d(self.query_dict["el"]["snd_timestamp_utc"],
                                   (self.query_dict["el"]["actualVelocity"].rolling(10).mean()),
                                   bounds_error=False)
            slew_velocity_az=interp1d(self.query_dict["az"]["snd_timestamp_utc"],
                                   (self.query_dict["az"]["actualVelocity"].rolling(10).mean()),
                                   bounds_error=False)

            slew_position=interp1d(self.query_dict["el"]["snd_timestamp_utc"],
                                   (self.query_dict["el"]["actualPosition"].rolling(10).mean()),
                                   bounds_error=False)

            sel=(slew_speed_el(overall_frame["times"]) > self.slew_speed_min) 
            sel |= (slew_speed_az(overall_frame["times"]) > self.slew_speed_min)
            overall_frame.loc[sel,"slew_state"]=True
            overall_frame["elevation_velocity"]=slew_velocity_el(overall_frame["times"])
            overall_frame["azimuth_velocity"]=slew_velocity_az(overall_frame["times"])
            overall_frame["elevation_position"]=slew_position(overall_frame["times"])
            overall_frame=overall_frame.loc[overall_frame["slew_state"]==True,:]
            
            if len(overall_frame) > 0:
                overall_frame["seq_num"]=query_dict["seq_num"]
                overall_frame["day_obs"]=query_dict["day_obs"]
                return overall_frame
            else: 
                return None
    async def run(self, dayObs):
        # given hp data iterate over all and create a dict with identified peaks as well as their height
        
        self.slews= await self.get_slews(dayObs)
        # make query
        client = makeEfdClient()
        event_list=[]
        for slew in tqdm(self.slews):
            await self.get_data(self.slews[0], client)
            np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
            result=self.identify(self.query_dict)
            if result is not None:
                event_list.append(result)
        if len(event_list) > 0:
            events_frame=pd.concat(event_list)
            return events_frame
        else:
            return None
        
        

        
            
            
        
        


        
if __name__ == '__main__':
    # want to understand if force actuators are on
    if not os.path.exists("./data/"):
        os.makedirs("./data/")
    
    begin_day_obs = 20230628
    end_day_obs = 20230628
    
    id_oscillations=identify_oscillation_events()
    
    current_day_obs=begin_day_obs
    while int(current_day_obs) <= int(end_day_obs):
        next_day_obs=calcNextDay(current_day_obs)
        print(current_day_obs)
        save_string=f"./data/oscillation_events_{current_day_obs}.csv" 
        if os.path.exists(save_string):
            print(f"file exists: {save_string}")
            current_day_obs=next_day_obs
            continue
        oscillation_events_frame = asyncio.run(id_oscillations.run(current_day_obs))
        if oscillation_events_frame is not None:
            oscillation_events_frame.to_csv(save_string)
            print("finished")

        current_day_obs=next_day_obs
        
        
        #
        # 
        #     start_time=end_time
        #     continue
        #print(f"starting query for {start_time} to {end_time} ")
        #await eventMaker.client.influx_client.close()
        #start_date = Time("2023-06-16T00:00:00", scale='utc')
    
        # overall_frame=asyncio.run(id_oscillations.run(start_time,end_time)) 
        # 
        # else:
        #     print(f"no data for {save_string} ")
        # start_time=end_time
        
        
        #print(overall_frame)
