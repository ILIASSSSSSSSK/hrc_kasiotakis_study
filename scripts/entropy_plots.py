import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
method_1_data=[
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/entropy.csv",    
]
method_2_data=[
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_2/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_1/data/entropy.csv",
]

method_3_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_25/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_26/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_27/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_28/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_29/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_30/data/entropy.csv"]

method_4_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_31/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_32/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_33/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_34/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_35/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_36/data/entropy.csv"]

method_5_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_41/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_37/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_38/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_39/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_40/data/entropy.csv"]

method_6_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_42/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_43/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_44/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_45/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_46/data/entropy.csv"]

method_7_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_50/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_49/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_48/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_47/data/entropy.csv",]

method_8_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_51/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_52/data/entropy.csv","/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_54/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_55/data/entropy.csv",]

method_9_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_60/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_57/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_58/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_59/data/entropy.csv",]

real_robot_method_7=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_56/data/entropy.csv","/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_56/data/entropy.csv"]

def plot_a_tempereture(method):
	if len(method)==1:
		df=pd.read_csv(method[0])
		temp=df["temp"]
		steps=range(0,len(temp))
		plt.plot(steps, temp, color='tab:blue', label='stim')
		plt.xlabel("#steps")
		plt.ylabel("temp")
		plt.title('tempreture hyperparameter per step')
		plt.grid(True)
		plt.ylim(0,1)
		plt.show()
	else:
		temp=[]
		len_t1=0
		len_t2=0
		for i in method:
			df=pd.read_csv(i)
			t=df["temp"].to_list()
			print(len(t))
			temp.append(t)
			print(len(t))
		print(len(temp))
		
		avg_temp=np.mean(np.array(temp), axis=0)
		std_temp = np.std(np.array(temp), axis=0)
		
		steps=range(0,len(avg_temp))
		plt.plot(steps, avg_temp, color='tab:blue', label='stim')
		plt.fill_between(steps, avg_temp - std_temp, avg_temp + std_temp, color='tab:blue', alpha=0.3)
		plt.xlabel("#steps")
		plt.ylabel("temp")
		plt.title('tempreture hyperparameter per step')
		plt.grid(True)
		plt.ylim(0,1)
		plt.show()
csvf=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_60/data/entropy.csv"]
plot_a_tempereture(csvf)
plot_a_tempereture(method_1_data)
