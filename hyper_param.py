# Train Settings
max_num_epoc = 1000
max_num_steps = 120 # 1000
plot_interval = 100
save_interval = 10 # 25
save_start_epoc = 0
train_time_per_epoc = 1000
save_best_start_epoc = 50
skip_epoc_num = 10 # 前几回合采用初始策略收集数据

# Gradient Projection Settings
gradient_projection_anealing_start_epoc = 200
gradient_projection_anealing_end_epoc = 500

# DAGGER Setting 
dagger_start_epoc = 50 # start to learn
dagger_ratio = 0.5 # 0.5, 1.0

# Evaluation Settings
evaluation_start_epoc = 0
evaluation_interval = 5 # 5 2
evaluation_epoc = 15 # 5 15
static_evaluation_interval = 1
static_evaluation_batch_size = 100

# Environment Settings
screen_width = 128
screen_height = 256
view = 60
decision_variable = 4
decision_freq = 1
bubble_width = 7
bubble_length = 20

# Replay Buffer Settings
buffer_size = 50000

# Learning Settings
batch_size = 30
learning_rate = 0.00001 # 0.00025、0.00005
discount_factor = 0.9

# input setting
reference_info_num = 12
traffic_direction_num = 16
waypoints_num = 30
vehicle_obs_distance = 50
center_grid_length = 10 # m
first_grid_length = 10
first_grid_time_gap = 1
# history_frame_num = 4
# feature_num = 5 # s, l, frenet_heading, velocity, index
# state_shape = (21, feature_num, history_frame_num)
history_frame_num = 1
feature_num = 5 # s, l, frenet_heading, velocity, index
state_shape = (21, feature_num, history_frame_num)

# epistemic_estimation
epistemic_estimation_freq = 1000 # 100
integrated_model_num = 6
lon_accuracy_threshold = 0.5

data_feature_file = '/home/szt/Dropbox/IDP/offline_rl/data/pre_process_output_202401060251/data_features.npy'