from datetime import datetime
import pandas as pd

motion_dict={}
# 230627
motion_dict[20230627]=pd.DataFrame([(datetime(2023,6,27,21,0),  5),
                                  (datetime(2023,6,28,0,3),5),
                                  
                                  (datetime(2023,6,28,0,53),10),
                                  (datetime(2023,6,28,1,42),5),
                                  (datetime(2023,6,28,2,16),6),
                                  (datetime(2023,6,28,3,22),7),
                                  (datetime(2023,6,28,4,15),8),
                                  (datetime(2023,6,28,5,3),10),
                                  (datetime(2023,6,28,6,35),10),
                                  (datetime(2023,6,28,9,26),20),
                                  (datetime(2023,6,28,10,12),5)
                                ], columns=["time", "motion_percent"])


# 230628
motion_dict[20230628]=pd.DataFrame([(datetime(2023,6,28,15,0),5),
                                  (datetime(2023,6,29,1,20), 10),
                                  (datetime(2023,6,29,2,23), 20),
                                  (datetime(2023,6,29,2,47), 10),
                                  (datetime(2023,6,29,3,16), 20),
                                  (datetime(2023,6,29,5,7), 25),
                                  (datetime(2023,6,29,7,20), 30),
                                  (datetime(2023,6,29,8,57), 40),
                                ], columns=[ "time","motion_percent"])
# should be run from /projects/tma directory
for key in motion_dict.keys():
    motion_dict[key].to_csv(f"./data/slew_configs/motion_settings_{key}.csv", index=False)
    
combined_frame=pd.concat([motion_dict[key] for key in motion_dict.keys()])
combined_frame.to_csv("./data/slew_configs/combined_motion_settings.csv", index=False)