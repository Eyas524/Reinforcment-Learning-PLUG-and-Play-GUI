import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QSpacerItem,
    QSizePolicy, QComboBox, QFrame
)


from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import QSize, Qt, QTimer, QDateTime

from PPOConfig import PPOConfig  # Assuming PPOConfig class
from SACConfig import SACConfig # Import SAC Config class
from DDPGConfig import DDPGConfig # Add class DDPG
from TRPOConfig import TRPOConfig 

from RoboDK_env import RoboDKEnv
#from Robot_train import RobotTrain


# From Robot_train:
import gym

from PPOConfig import PPOConfig
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from sb3_contrib import TRPO
from stable_baselines3 import SAC

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv

from Robodk_gui import RoboDKgui
from RoboDK_env import RoboDKEnv
import torch
from stable_baselines3.common.env_util import make_vec_env
from StopCallBack import EarlyStoppingCallback



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.stop_training = False
        self.selectedWaypoints = ["", ""]

        self.setWindowTitle("RL-based plug-in GUI")
        self.setFixedSize(QSize(900, 1000))

        # === Main Vertical Layout ===
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # === Panel 1 with border ===
        panel1_frame = QFrame()
        #panel1_layout = self.create_top_bar()
        #panel1_frame.setLayout(panel1_layout)
        

        # === Create Horizontal Layout for Panel 2 and Panel 3 ===
        horizontal_panel_layout = QHBoxLayout()

        # === Panel 2 ===
        panel2_frame = QFrame()
        panel2_layout = self.panel_RL()
        panel2_frame.setLayout(panel2_layout)

        # === Panel 3 ===
        panel3_frame = QFrame()
        panel3_layout = self.panel_roboDK()
        panel3_frame.setLayout(panel3_layout)

        # === Add both panels to the horizontal layout ===
        horizontal_panel_layout.addWidget(panel2_frame)
        horizontal_panel_layout.addSpacing(10)  # Space between panels
        horizontal_panel_layout.addWidget(panel3_frame)
        
        # === Panel 4 ===
        panel4_frame = QFrame()
        panel4_layout = self.panel_commands()
        panel4_frame.setLayout(panel4_layout)

        # === Add horizontal layout to main vertical layout ===
        main_layout.addWidget(panel1_frame, alignment=Qt.AlignLeft | Qt.AlignTop)
        main_layout.addLayout(horizontal_panel_layout)
        main_layout.addWidget(panel4_frame, alignment=Qt.AlignLeft)

        # === Panel 4: Hyperparameters ===
        self.hyperparam_layout = QVBoxLayout()
        main_layout.addLayout(self.hyperparam_layout)

        
        # === Spacer to push everything up ===
        main_layout.addStretch()



    def update_time(self):
        current_time = QDateTime.currentDateTime().toString("hh:mm:ss AP")
        self.time_label.setText(current_time)

    # ---------- PANEL 2 ----------

    def panel_RL(self):
        layout = QVBoxLayout()

        # Spacer to align with the connection button in Panel 3
        layout.addSpacing(7)

        label = QLabel("RL-Algorithms")
        label.setStyleSheet("font-weight: bold; padding: 5px;")
        label.setFixedSize(QSize(150, 30))  # Match label height to connect button
        layout.addWidget(label)

        rl_dropdown = self.list_RL()
        rl_dropdown.setFixedHeight(30)  # Match height with other combo box
        layout.addWidget(rl_dropdown)

        return layout


    def list_RL(self):

        self.rl_combo = QComboBox()
        self.rl_combo.addItems([
            "Select algorithm",
            "Proximal Policy Optimization (PPO)",
            "Soft Actor-Critic (SAC)",
            "Deep Deterministic Policy Gradient (DDPG)",
            "Trust Region Policy Optimization (TRPO)"
        ])
        self.rl_combo.setFixedWidth(350)
        self.rl_combo.currentIndexChanged.connect(self.update_hyperparam_panel)
        return self.rl_combo


    def update_hyperparam_panel(self):
        
        algo = self.rl_combo.currentText()
        print(algo)
        # Clear existing panel
        for i in reversed(range(self.hyperparam_layout.count())):
            widget = self.hyperparam_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Load the correct panel
        if algo.startswith("Proximal"):
            self.hyperparam_layout.addWidget(PPOConfig())

        if algo.startswith("Soft"):
            self.hyperparam_layout.addWidget(SACConfig())

        if algo.startswith("Deep"):
            self.hyperparam_layout.addWidget(DDPGConfig())

        if algo.startswith("Trust"):
            self.hyperparam_layout.addWidget(TRPOConfig())

        
#-----------------------------Panel 3-----------------RoboDK panel------------------------------------------
    def panel_roboDK(self):
        layout = QVBoxLayout()

        # === 1. Connection Row ===
        connection_layout = self.create_connection_section()
        layout.addLayout(connection_layout)
        layout.addSpacing(5)

        # === 2. Waypoint Selector ===
        waypoint_layout = self.create_waypoint_selector()
        layout.addLayout(waypoint_layout)
        layout.addSpacing(10)

        # === 3. Start/End Waypoint Labels ===
        display_layout = self.create_selected_waypoints_display()
        layout.addLayout(display_layout)

        return layout

    def create_connection_section(self):
        layout = QHBoxLayout()

        self.connect_btn = QPushButton("Connect to RoboDK")
        self.connect_btn.setFixedWidth(200)
        self.connect_btn.clicked.connect(self.connect_to_robodk)
        layout.addWidget(self.connect_btn, alignment=Qt.AlignLeft)
        
        self.status_label = QLabel()
        self.status_label.setFixedSize(QSize(100, 20))
        self.status_label.setText("Status: ")
        layout.addWidget(self.status_label)

        self.status_light = QLabel()
        self.status_light.setFixedSize(20, 20)
        self.status_light.setStyleSheet("background-color: red; border: 1px solid black;")
        
        layout.addWidget(self.status_light)

        return layout

    def create_waypoint_selector(self):
        layout = QHBoxLayout()

        self.point_dropdown = QComboBox()
        self.point_dropdown.setFixedWidth(200)
        layout.addWidget(self.point_dropdown)

        self.set_start_btn = QPushButton("Set Start Waypoint")
        self.set_start_btn.clicked.connect(self.set_start_waypoint)
        layout.addWidget(self.set_start_btn)

        self.set_end_btn = QPushButton("Set End Waypoint")
        self.set_end_btn.clicked.connect(self.set_end_waypoint)
        layout.addWidget(self.set_end_btn)

        return layout

    def create_selected_waypoints_display(self):
        layout = QVBoxLayout()

        self.start_label = QLabel("Start waypoint : ")

        self.end_label = QLabel("End waypoint  : ")

        layout.addWidget(self.start_label)
        layout.addWidget(self.end_label)

        return layout

    # ----------Placeholder logic for connection----------
    def connect_to_robodk(self):
        env = RoboDKEnv()
        self.status_light.setStyleSheet("background-color: green; border: 1px solid black;")
        self.point_dropdown.clear()
        # Simulated list of points
        points = []
        targets = env.getTargets()
        for t in targets:
            points.append(t.Name())

        #demo_points = ["Point A", "Point B", "Point C", "Point D"]
        self.point_dropdown.addItems(points)

    def set_start_waypoint(self):
        selected = self.point_dropdown.currentText()
        self.start_label.setText(f"Start waypoint: {selected}")
        self.selectedWaypoints[0] = selected
        print(self.selectedWaypoints[0])

    def set_end_waypoint(self):
        selected = self.point_dropdown.currentText()
        self.end_label.setText(f"End waypoint: {selected}")
        self.selectedWaypoints[1] = selected

#-----------------------------Panel 4-----------------Start and Stop panel------------------------------------------
    def panel_commands(self):
        layout = QHBoxLayout()

        self.start_train = QPushButton("Start training")
        self.start_train.setFixedSize(QSize(150, 40))
        self.start_train.clicked.connect(self.set_start_training)

        self.stop_train = QPushButton("Stop training")
        self.stop_train.setFixedSize(QSize(150, 40))
        self.stop_train.clicked.connect(self.set_stop_training)
        

        layout.addWidget(self.start_train)

        layout.addWidget(self.stop_train)

        #layout.addWidget(self.save_model)

        return layout

    def set_start_training(self):
        print("start training is preseed")
        

        self.raw_env = RoboDKgui(start_waypoint=self.selectedWaypoints[0], end_waypoint=self.selectedWaypoints[1])

        #check_env(self.raw_env)

        self.env_monitor = Monitor(self.raw_env) 

        self.env = DummyVecEnv([lambda: self.env_monitor])

        self.eval_env = DummyVecEnv([lambda: Monitor(RoboDKgui( start_waypoint=self.selectedWaypoints[0], end_waypoint=self.selectedWaypoints[1]))])


        
        eval_callback = EvalCallback(self.eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=5000, n_eval_episodes=5)
        

        early_stop = EarlyStoppingCallback(gui_instance=self)


        # Check if GPU is available and use it
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        algo = self.rl_combo.currentText()

        if algo.startswith("Proximal"):
            for i in reversed(range(self.hyperparam_layout.count())):
                widget = self.hyperparam_layout.itemAt(i).widget()

                self.policy_value                = widget.get_policy_value()
                self.verbose_value               = widget.get_verbose_value()
                self.learning_rate_value         = widget.get_learning_rate_value()
                self.n_steps_value               = widget.get_n_steps_value()
                self.batch_size                  = widget.get_batch_size_value()
                self.n_epochs_value              = widget.get_n_epochs_value()
                self.gamma_value                 = widget.get_gamma_value()
                self.clip_range_value            = widget.get_clip_range_value()
                self.clip_range_vf_value         = widget.get_clip_range_vf_value()
                self.normalize_advantage         = np.bool_(widget.get_normalize_advantage_value().strip().lower() == "true")
                self.ent_coef_value              = widget.get_ent_coef_value()
                self.vf_coef_value               = widget.get_vf_coef_value()
                self.max_grad_norm_value         = widget.get_max_grad_norm_value()
                self.gae_lambda_value            = widget.get_gae_lambda_value()
                self.use_sde_value               = np.bool_(widget.get_use_sde_value().strip().lower() == "true")
                self.sde_sample_freq_value       = widget.get_sde_sample_freq_value()
                self.rollout_buffer_class_value  = widget.get_rollout_buffer_class_value()
                self.rollout_buffer_kwargs_value = widget.get_rollout_buffer_kwargs_value()
                self.init_setup_model_value      = np.bool_(widget.get_init_setup_model_value().strip().lower()=="true")
                self.target_kl_value             = widget.get_target_kl_value()
                self.stats_window_size           = widget.get_stats_window_size_value()
                self.tensorboard_log_value       = widget.get_tensorboard_log_value()
                self.policy_kwargs_value         = widget.get_policy_kwargs_value()
                self.seed_value                  = widget.get_seed_value()
                self.device_value                = widget.get_device_value()
                print('model declaration')


                print("-------------------------------------------------------")
                print("policy:",                        self.policy_value)
                print("verbose:",                       self.verbose_value)
                print("Learning Rate:",                 self.learning_rate_value)
                print("n_steps:",                       self.n_steps_value)
                print("batch size:",                    self.batch_size)
                print("n_epochs:",                      self.n_epochs_value)
                print("gamma:",                         self.gamma_value)
                print("clip_range:",                    self.clip_range_value)
                print("clip_range_vf:",                 self.clip_range_vf_value)
                print("normalize_advantage:",           self.normalize_advantage)
                print("ent_coef:",                      self.ent_coef_value)
                print("vf_coef:",                       self.vf_coef_value)
                print("max_grad_norm:",                 self.max_grad_norm_value)
                print("gae_lambda:",                    self.gae_lambda_value)
                print("use_sde:",                       self.use_sde_value)
                print("sde_sample_freq:",               self.sde_sample_freq_value)
                print("rollout_buffer_class_value:",    self.rollout_buffer_class_value)
                print("rollout_buffer_kwargs_value:",   self.rollout_buffer_kwargs_value)
                print("init_setup_model_value:",        self.init_setup_model_value)
                print("target_kl:",                     self.target_kl_value)
                print("stats_window_size:",             self.stats_window_size)
                print("tensorboard_log:",               self.tensorboard_log_value)
                print("policy_kwargs:",                 self.policy_kwargs_value)
                print("seed:",                          self.seed_value)
                print("device:",                        self.device_value)
                print("-------------------------------------------------------")
                print("Model declaration is done")


            model = PPO(policy= self.policy_value,
                        env = self.env,
                        learning_rate=float(self.learning_rate_value),
                        n_steps=int(self.n_steps_value),
                        batch_size=int(self.batch_size),
                        n_epochs=int(self.n_epochs_value),
                        gamma=float(self.gamma_value),
                        gae_lambda=float(self.gamma_value),
                        clip_range=float(self.clip_range_value),
                        clip_range_vf= None,
                        normalize_advantage= self.normalize_advantage,
                        ent_coef= float(self.ent_coef_value),
                        vf_coef= float(self.vf_coef_value),
                        max_grad_norm= float(self.max_grad_norm_value),
                        use_sde= self.use_sde_value,
                        sde_sample_freq= int(self.sde_sample_freq_value),
                        rollout_buffer_class= None,
                        rollout_buffer_kwargs= None,
                        target_kl= None,
                        stats_window_size= int(self.stats_window_size),
                        tensorboard_log= None,
                        policy_kwargs= None,
                        verbose= int(self.verbose_value),
                        seed= None,
                        device= self.device_value,
                        _init_setup_model= self.init_setup_model_value)

            # Train the agent with the callback
            print('Learning the model')
            model.learn(total_timesteps=5000, callback=[eval_callback, early_stop])

            print('Saving the model')
            # Save the model
            model.save("ppo_ur10_robodk")
            del model

            model.load("ppo_ur10_robodk")

            # Test the trained model
            print('Testing the model...')
            obs, _ = self.env.reset()
            for _ in range(1000):
                action, _states = model.predict(obs)
                obs, rewards, dones, truncated, info = self.env.step(action)
                if dones or truncated:
                    print("Breaking the program")
                    break


        if algo.startswith("Deep"):
            for i in reversed(range(self.hyperparam_layout.count())):
                widget = self.hyperparam_layout.itemAt(i).widget()

                self.policy_value                           = widget.get_policy_value()
                self.gamma_value                            = widget.get_gamma_value()
                self.action_noise_value                     = widget.get_action_noise_value()
                self.buffer_size_value                      = widget.get_buffer_size_value()
                self.batch_size_value                       = widget.get_batch_size_value()
                self.tau_value                              = widget.get_tau_value()
                self.verbose_value                          = widget.get_verbose_value()
                self.policy_kwargs_value                    = widget.get_policy_kwargs_value()
                self.seed_value                             = widget.get_seed_value()
                self.tensorboard_log_value                  = widget.get_tensorboard_log_value()
                self.init_setup_model_value                 = widget.get_init_setup_model_value()


                print('Start the model declaration')

                print("-------------------------------------------------------")
                print("policy:",                        self.policy_value)
                print("gamma:",                         self.gamma_value )
                print("action noise:",                  self.action_noise_value)
                print("buffer size:",                   self.buffer_size_value)
                print("batch size:",                    self.batch_size_value)
                print("tau:",                           self.tau_value)
                print("verbose:",                       self.verbose_value)
                print("policy kwargs:",                 self.policy_kwargs_value)
                print("seed :",                         self.seed_value)
                print("tensorboard log:",               self.tensorboard_log_value)
                print("init setup model:",              self.init_setup_model_value)
                
                print("-------------------------------------------------------")

                print("Model declaration is done")


            model = DDPG(policy = self.policy_value ,
                         env = self.env,
                         gamma= self.gamma_value ,
                         action_noise=None,
                         tau = self.tau_value , 
                         batch_size= self.batch_size_value,
                         buffer_size= self.buffer_size_value,
                         verbose=  self.verbose_value,
                         tensorboard_log=None,
                         _init_setup_model=True, 
                         policy_kwargs=None,
                         seed=None)

            # Train the agent with the callback
            print('Learning the model...')
            model.learn(total_timesteps=400000, callback=[eval_callback])

            print('Saving the model...')
            # Save the model
            model.save("DDPG_ur10_robodk")

            del model
            model = DDPG.load("DDPG_ur10_robodk")

            # Test the trained model
            print('Testing the model')
            obs, _ = self.env.reset()
            for _ in range(1000):
                action, _states = model.predict(obs)
                obs, rewards, dones, truncated, info = self.env.step(action)
                if dones or truncated:
                    print("Breaking the program")
                    break
        
        
        if algo.startswith("Soft"):
            print("Training started with the training of SAC algorithm")

            for i in reversed(range(self.hyperparam_layout.count())):
                widget = self.hyperparam_layout.itemAt(i).widget()

                self.policy_value                           = widget.get_policy_value()
                self.gamma_value                            = widget.get_gamma_value()
                self.learning_rate_value                    = widget.get_learning_rate_value()
                self.buffer_size_value                      = widget.get_buffer_size_value()
                self.learning_starts_value                  = widget.get_learning_starts_value()
                self.train_freq_value                       = widget.get_train_freq_value()
                self.batch_size_value                       = widget.get_batch_size_value()
                self.tau_value                              = widget.get_tau_value()
                self.ent_coef_value                         = widget.get_ent_coef_value()
                self.target_update_interval_value           = widget.get_target_update_interval_value()
                self.gradient_steps_value                   = widget.get_gradient_steps_value()
                self.target_entropy_value                   = widget.get_target_entropy_value()
                self.action_noise_value                     = widget.get_action_noise_value()
                self.verbose_value                          = widget.get_verbose_value()
                self.tensorboard_log_value                  = widget.get_tensorboard_log_value()
                self.init_setup_model_value                 = widget.get_init_setup_model_value()
                self.policy_kwargs_value                    = widget.get_policy_kwargs_value()
                self.seed_value                             = widget.get_seed_value()
                
                print('Start the model declaration')

                print("-------------------------------------------------------")
                print("policy:",                                self.policy_value)
                print("gamma:",                                 self.gamma_value)
                print("learning rate:",                         self.learning_rate_value)
                print("buffer size:",                           self.buffer_size_value)
                print("learning starts:",                       self.learning_starts_value)
                print("train freq:",                            self.train_freq_value)
                print("batch size:",                            self.batch_size_value)
                print("tau value:",                             self.tau_value)
                print("ent coef:",                              self.ent_coef_value)
                print("target update interval:",                self.target_update_interval_value)
                print("gradient steps:",                        self.gradient_steps_value)                   
                print("target entropy:",                        self.target_entropy_value)
                print("action noise:",                          self.action_noise_value)
                print("verbose:",                               self.verbose_value)
                print("tensorboard log:",                       self.tensorboard_log_value)
                print("init setup model:",                      self.init_setup_model_value)         
                print("policy kwargs:",                         self.policy_kwargs_value)
                print("seed :",                                 self.seed_value)

                print("-------------------------------------------------------")

                print("Model declaration is done")

            model = SAC(policy                  = self.policy_value,
                        env                     = self.env, 
                        gamma                   = self.gamma_value,
                        learning_rate           = self.learning_rate_value,
                        buffer_size             = self.buffer_size_value, 
                        learning_starts         = self.learning_starts_value,
                        train_freq              = self.train_freq_value,
                        batch_size              = self.batch_size_value,
                        tau                     = self.tau_value, 
                        ent_coef                = self.ent_coef_value,
                        target_update_interval  = self.target_update_interval_value,
                        gradient_steps          = self.gradient_steps_value, 
                        target_entropy          = self.target_entropy_value,
                        action_noise            = None,
                        verbose                 = self.verbose_value,
                        tensorboard_log         = None, 
                        _init_setup_model       = self.init_setup_model_value,
                        policy_kwargs           = None,
                        seed                    = None)
            
            print("Learning the model")
            model.learn(total_timesteps=50000, callback = [eval_callback])

            print("Saving the model")

            model.save("SAC_ur10_robodk")

            del model

            model.load("SAC_ur10_robodk")

            #Test the trainied model:
            print("Testing the model")

            obs, _ = self.env.reset()
            for _ in range(1000):
                action, _states = model.predict(obs)
                obs, rewards, dones, truncated, info = self.env.step(action)
                if dones or truncated:
                    print("Breaking the program")
                    break


        if algo.startswith("Trust"):
            print("training started with the training of TRPO algorithm")

            for i in reversed(range(self.hyperparam_layout.count())):
                widget = self.hyperparam_layout.itemAt(i).widget()

                self.policy_value                                   = widget.get_policy_value()
                self.learning_rate_value                            = widget.get_learning_rate_value()
                self.gamma_value                                    = widget.get_gamma_value()
                self.timesteps_per_batch_value                      = widget.get_timesteps_per_batch_value()
                self.max_kl_value                                   = widget.get_max_kl_value()
                self.cg_iters_value                                 = widget.get_cg_iters_value()
                self.lam_value                                      = widget.get_lam_value()
                self.entcoeff_value                                 = widget.get_entcoeff_value()
                self.cg_damping_value                               = widget.get_cg_damping_value()
                self.vf_stepsize_value                              = widget.get_vf_stepsize_value()
                self.vf_iters_value                                 = widget.get_vf_iters_value()
                self.verbose_value                                  = widget.get_verbose_value()
                self.tensorboard_log_value                          = widget.get_tensorboard_log_value()
                self.init_setup_model_value                         = widget.get_init_setup_model_value()
                self.policy_kwargs_value                            = widget.get_policy_kwargs_value()
                self.full_tensorboard_log_value                     = widget.get_full_tensorboard_log_value()
                self.seed_value                                     = widget.get_seed_value()
                self.n_cpu_tf_sess_value                            = widget.get_n_cpu_tf_sess_value()
                
                print('Start the model declaration')

                print("-------------------------------------------------------")
                print("policy:",                                self.policy_value)
                print("gamma:",                                 self.gamma_value)
                print("cg damping:",                            self.cg_damping_value)
                print("verbose:",                               self.verbose_value)
                print("init setup model:",                      self.init_setup_model_value)
                print("-------------------------------------------------------")

                print("Model declaration is done")


            model = TRPO(policy               = self.policy_value,
                        env                     = self.env, 
                        gamma                   = self.gamma_value,
                        cg_damping              = self.cg_damping_value, 
                        verbose                 = self.verbose_value,
                        tensorboard_log         = None, 
                        _init_setup_model       = self.init_setup_model_value,
                        policy_kwargs=None,
                        seed=None,
                        )
            
            print("Learning the model")
            model.learn(total_timesteps=50000, callback = [eval_callback])

            print("Saving the model...")

            model.save("TRPO_ur10_robodk")

            del model

            model.load("TRPO_ur10_robodk")

            #Test the trainied model:
            print("Testing the model")

            obs, _ = self.env.reset()
            for _ in range(1000):
                action, _states = model.predict(obs)
                obs, rewards, dones, truncated, info = self.env.step(action)
                if dones or truncated:
                    print("Breaking the program")
                    break

  #################################################################          


        #not implemented yet CTR + C to stop training.
    def set_stop_training(self):
        print("Stop training is pressed")
        self.stop_training = True
        


# ---------- MAIN ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
