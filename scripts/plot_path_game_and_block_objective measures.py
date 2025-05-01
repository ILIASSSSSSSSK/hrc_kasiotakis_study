import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
#select the experiment 
file_rl_data=[#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_1/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_2/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/rl_test_data.csv"
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_16/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_19/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_20/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_31/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_32/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_33/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_34/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_35/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_36/data/rl_test_data.csv",
]
file_data=[#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_1/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_2/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/test_data.csv"
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_16/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_19/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_20/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_31/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_32/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_33/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_34/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_35/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_36/data/test_data.csv",
]
#plot graph of trajectory for selected game and block

def plot_path_graph(files,block=2,game=10):
 num_games= len(files)
 rows,cols=0,0
 if len(files)==1:
 	rows,cols=1,1
 else:
 	if (len(files)==3) or (len(files)==2):
 		rows,cols=1,len(files)
 	elif(len(files)==4):
 		rows,cols=2,2
 	elif(len(files)==7):
 		rows=3
 		cols=3
 	else:
 		if len(files)%2==0:
 			cols=int(len(files)/2)
 			rows=cols
 		else:
 			cols=int(len(files)/2)
 			rows=int(len(files)/2)+1
 			if len(files)==9:
 				cols=3
 				rows=3

 fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
 if (cols==1) and (rows==1):
 	axes=[axes]
 else:
 	axes = axes.flatten()
 

 #select block (consider thet you start from 1 (ex 2nd block: 2))
 block=block
 #select the game you want to plot (consider thet you start from 1 (ex 2nd game: 2)) 
 game=game
 for j,file in enumerate(files):
 	df=pd.read_csv(file)
 	ee_pos_x_prev="ee_pos_x_prev" 
 	ee_pos_y_prev="ee_pos_y_prev"
 	ee_pos_x_next="ee_pos_x_next"
 	ee_pos_y_next="ee_pos_y_next"
 	g=0
 	all_games_indexes=[0]
 	positions_x=[]
 	positions_y=[]
 	for i in range(len(df[ee_pos_y_prev])):
 		if (df[ee_pos_x_prev][i]==0.0)and(df[ee_pos_x_next][i]==0.0)and(df[ee_pos_y_prev][i]==0.0)and(df[ee_pos_y_next][i]==0.0):
 			all_games_indexes.append(i)
 	game_id=(block-1)*10+(game-1)
 	for i in range(all_games_indexes[game_id]+1,all_games_indexes[game_id+1]):
 		positions_x.append(df[ee_pos_x_prev][i])
 		positions_y.append(df[ee_pos_y_prev][i])
 		if i==(all_games_indexes[game_id+1]-1):
 			positions_x.append(df[ee_pos_x_next][i])
 			positions_y.append(df[ee_pos_y_next][i])
 	goal=[-0.265, 0.251]
 	goal_distance=0.01
 	#the first point is the final of the last game so ignore it (start from 1, not 0)
 	ax=axes[j]
 	
 	line=Line2D(positions_x[1:],positions_y[1:],color='blue')
 	ax.add_line(line)
 	#ax.scatter(positions_x[1:],positions_y[1:],color='blue',label='Intermediate Points')
 	ax.scatter(positions_x[1],positions_y[1],color='pink',label='Starting Point')
 	ax.scatter(positions_x[-1],positions_y[-1],color='green',label='Final Point')
 	ax.scatter(-0.346, 0.333,color='red')
 	ax.scatter(-0.345, 0.172,color='red')
 	ax.scatter(-0.185, 0.332,color='red')
 	ax.scatter(-0.184, 0.172,color='red')
 	ax.scatter(goal[0],goal[1],color='purple',label='Goal')
 	circle=patches.Circle(goal,goal_distance,color='purple',fill=False,linewidth=1)
 	ax.add_patch(circle)
 	ax.set_aspect('equal')
 	ax.set_xlabel('X')
 	ax.set_ylabel('Y')
 	ax.legend(fontsize="5",loc='upper right')
 	title=file.replace("/data/rl_test_data.csv","")
 	title2=""
 	if "expert" in title:
 		title2="Expert Experiment #"+title[-2]+title[-1]
 	elif "LfD_TL" in file:
 		title2="TL Experiment #"+title[-2]+title[-1]
 	elif "LfD_no_TL" in file:
 		title2="no_TL Experiment #"+title[-2]+title[-1]


 	ax.set_title(title2+(' Path for game %i in block %i'%(game,block)))
 	ax.set_xlim(-0.38,0)
 	ax.set_ylim(0.13,0.35)
 # Hide extra subplots (if any)
 for j in range(i + 1, len(axes)):
 	fig.delaxes(axes[j])

 # Adjust layout to prevent overlap
 
 plt.tight_layout()
 plt.show()

def plot_wins(files):
	#calculate wins for each block
	wins_per_file=[]
	for file in files:
		df1=pd.read_csv(file)
		game_rew=df1["Rewards"]
		wins_per_block=[]

		for i in range(0,len(game_rew),10):
			wins=0
			for j in range(10):
				if game_rew[j+i]!=(-150):
					wins+=1
			wins_per_block.append(wins)
		
		wins_per_file.append(wins_per_block)
	
	wins_block_diagram=[]
	blocks=[]
	wins=np.zeros((len(wins_per_file),8))
	for i in range(len(wins_per_file)):
		for j in range(8):
			wins[i][j]=wins_per_file[i][j]
			wins_block_diagram.append(wins[i,j])
			blocks.append(j)            
	df2=pd.DataFrame({'Wins': wins_block_diagram, 'Blocks': blocks})
	
	avg_wins=np.mean(wins, axis=0)
	std_wins = np.std(wins, axis=0)
	print("Average Wins: ")
	print(avg_wins)
	print("Std Wins: ")
	print(std_wins)
	blocks=range(0,8)
	plt.plot(blocks, avg_wins, color='tab:blue', label='stim')
	plt.fill_between(blocks, avg_wins - std_wins, avg_wins + std_wins, color='tab:blue', alpha=0.3)
	plt.xlabel("Block")
	plt.ylabel("#wins")
	plt.title('Wins per Block')
	plt.grid(True)
	plt.show()
	fig=fig = px.box(df2, x="Blocks", y="Wins",title="Wins per Block")
	plot(fig,auto_open=True)

def plot_rewards_and_norm_dist(files):
	#norm_dist=tot_dist*tot_time/max_time
	
	rewards_per_file=[]
	rew_blocks_for_all=[]
	dist_per_file=[]
	dur_per_file=[]
	norm_dist_block_for_all=[]
	for file in files:
		df1=pd.read_csv(file)
		game_rew=df1["Rewards"]
		tot_dist=df1["Travelled Distance"]
		tot_time=df1["Episodes Duration in Seconds"]
		rewards_per_file.extend(game_rew)
		dist_per_file.extend(tot_dist)
		dur_per_file.extend(tot_time)
		
	
	max_time = 30.0
	print("Max time in secs: ")
	print(max_time)
	
	for i in range(0,80,10):
		rew_per_bl=[]
		norm_dist_per_bl=[]
		for j in range(0,len(rewards_per_file),80):
			for l in range(0,10):
				rew_per_bl.append(rewards_per_file[l+i+j]+150)
				norm_dist_per_bl.append(dist_per_file[l+i+j]*dur_per_file[l+i+j]/max_time)
				
		rew_blocks_for_all.append(rew_per_bl)
		norm_dist_block_for_all.append(norm_dist_per_bl)
		
	avg_rewards=[]
	std_rewards=[]
	avg_norm_dist=[]
	std_norm_dist=[]
	if len(files)==1:
		for i in range(len(rew_blocks_for_all)):
			avg_r=np.mean(rew_blocks_for_all[i])
			std_r=np.std(rew_blocks_for_all[i])
			avg_nd=np.mean(norm_dist_block_for_all[i])
			std_nd=np.std(norm_dist_block_for_all[i])
			avg_rewards.append(avg_r)
			std_rewards.append(std_r)
			avg_norm_dist.append(avg_nd)
			std_norm_dist.append(std_nd)

	else:
		rewards_block_diagram=[]
		norm_dist_block_diagram=[]
		block=0
		block_id=[]
		for i in range(len(rew_blocks_for_all)):
			avgs=[]
			avgs_nd=[]
			for j in range(0,len(rew_blocks_for_all[0]),10):								
				avgs.append(np.mean(rew_blocks_for_all[i][j:j+10]))
				avgs_nd.append(np.mean(norm_dist_block_for_all[i][j:j+10]))
				rewards_block_diagram.append(np.mean(rew_blocks_for_all[i][j:j+10]))
				norm_dist_block_diagram.append(np.mean(norm_dist_block_for_all[i][j:j+10]))
				block_id.append(block)
		    
			avg_rewards.append(np.mean(avgs))
			std_rewards.append(np.std(avgs))
			avg_norm_dist.append(np.mean(avgs_nd))
			std_norm_dist.append(np.std(avgs_nd))
			block+=1
		df3=pd.DataFrame({'Rewards': rewards_block_diagram, 'Blocks': block_id})
		df4=pd.DataFrame({'Norm_Dist': norm_dist_block_diagram, 'Blocks': block_id})
		fig3=px.box(df3, x="Blocks", y="Rewards",title="Reward per Block")
		plot(fig3,filename='fig3.html',auto_open=True)
		fig2 = px.box(df4, x="Blocks", y="Norm_Dist",title="Normalized Distance per Block")
		plot(fig2,filename='fig2.html',auto_open=True)
			





	print("")
	print("Average rewards: ")
	print(avg_rewards)
	print('Std: ')
	print(std_rewards)
	print("")
	print("Average Normalised Distance: ")
	print(avg_norm_dist)
	print("Std Normalised Distance: ")
	print(std_norm_dist)


	blocks=range(0,8)
	plt.plot(blocks, avg_rewards, color='tab:blue', label='stim')
	plt.fill_between(blocks, np.array(avg_rewards) - np.array(std_rewards), np.array(avg_rewards) + np.array(std_rewards), color='tab:blue', alpha=0.3)
	plt.xlabel("Block")
	plt.ylabel("Reward")
	plt.title('Reward per Block')
	plt.grid(True)
	plt.show()

	plt.plot(blocks, avg_norm_dist, color='tab:blue', label='stim')
	plt.fill_between(blocks, np.array(avg_norm_dist) - np.array(std_norm_dist), np.array(avg_norm_dist) + np.array(std_norm_dist), color='tab:blue', alpha=0.3)
	plt.xlabel("Block")
	plt.ylabel("Normalised Distance")
	plt.title('Normalised Distance per Block')
	plt.grid(True)
	plt.show()

def plot_heatmap_with_coverage(batch_number, filepaths, steps_filepaths, games_per_batch=10, threshold=0, max_x=-0.18, min_x=-0.349, max_y=0.330, min_y=0.170, smoothing_sigma=0.3, ax=None):
    all_x_coords = []
    all_y_coords = []

    # Helper function to get game data
    def get_game_data(game_number, test_data, rl_data):
        if game_number < 0 or game_number >= len(test_data):
            raise ValueError("Invalid game number.")
        start_index = 0 if game_number == 0 else int(np.sum(test_data[:game_number, -1])) + game_number
        num_rows = int(test_data[game_number, -1])

        game_data = rl_data[start_index:start_index+num_rows, :]
        return game_data

    # Helper function to get batch data
    def get_batch_data(batch_number, test_data, rl_data, games_per_batch):
        start_game = batch_number * games_per_batch
        
        end_game = start_game + games_per_batch
        x_coords = []
        y_coords = []
        for game_num in range(start_game, end_game):
            game_data = get_game_data(game_num, test_data, rl_data)
            x_coords.extend(game_data[:, 2])
            y_coords.extend(game_data[:, 3])
        return x_coords, y_coords
    
    # Iterate over each participant
    for test_data, rl_data in zip(filepaths, steps_filepaths):
        x_coords, y_coords = get_batch_data(batch_number, test_data, rl_data, games_per_batch)
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)
    # Create a 2D histogram for the heatmap

    heatmap, xedges, yedges = np.histogram2d(all_x_coords, all_y_coords, bins=[np.linspace(min_x, max_x, 20), np.linspace(min_y, max_y, 20)])
    total_bins = np.prod(heatmap.shape)
    filled_bins = np.nansum(heatmap > 0)
    coverage = filled_bins / total_bins
    heatmap_normalized = heatmap / np.max(heatmap)

    # Apply Gaussian smoothing to the heatmap
    #smoothed_heatmap = gaussian_filter(heatmap, smoothing_sigma)
    smoothed_heatmap = gaussian_filter(heatmap_normalized, smoothing_sigma)

    # Plotting the smoothed heatmap
    if ax is None:
        plt.figure(figsize=(8, 6), facecolor='white')
    im=ax.imshow(smoothed_heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='YlGn', aspect='auto')
    #cbar = plt.colorbar(im, ax=ax)
    #cbar.set_label('Counts')
    if ax is None:
        plt.colorbar(label='Counts')
        plt.title(f"Smoothed Heatmap of Positions in Batch {batch_number} ")
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        #plt.gca().set_facecolor('black')
    


    print(f"Coverage for Batch {batch_number}: {coverage:.2%}")

    return coverage  # Optionally return the coverage value

expert_test_data=[]
expert_steps_test_data=[]

plot_path_graph(file_rl_data)
plot_wins(file_data)
plot_rewards_and_norm_dist(file_data)
for expert_test_data_file, expert_steps_test_data_file in zip(file_data, file_rl_data):
    expert_test_data.append(np.loadtxt(expert_test_data_file, delimiter=',', skiprows=1))
    expert_steps_test_data.append(np.loadtxt(expert_steps_test_data_file, delimiter=',', skiprows=1))
batches_to_collect = [0, 3, 5]
col_labels = ["Baseline", "Block 3", "Block 6"]
row_labels = ["Expert"]
num_rows=len(row_labels)
num_cols=len(col_labels)
fig = plt.figure(figsize=(13, 5))
for row in range(num_rows):
    for col in range(num_cols):
        subplot_idx = row * num_cols + col + 1
        ax = fig.add_subplot(num_rows, num_cols, subplot_idx)

        # Determine the group based on the row (0 for Experts, 1 for TL, 2 for No TL)
        group_idx = row

        # Get the batch number based on the column
        batch_number = batches_to_collect[col]

        # Construct a title based on the labels and batch number
        title = f"{row_labels[group_idx]} - {col_labels[col]}"

        if group_idx == 0:
            # Expert
            plot_heatmap_with_coverage(batch_number, expert_test_data, expert_steps_test_data, ax=ax)
        elif group_idx == 1:
            # TL Participant
            plot_heatmap_with_coverage(batch_number, TL_test_data, TL_steps_test_data, ax=ax)
        elif group_idx == 2:
            # No TL Participant
            plot_heatmap_with_coverage(batch_number, NO_TL_test_data, NO_TL_steps_test_data, ax=ax)
        

        ax.set_title(title)

# Adjust the layout
plt.tight_layout()

# Show the figure
plt.show()
