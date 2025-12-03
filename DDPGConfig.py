from PyQt5.QtWidgets import QWidget, QLineEdit, QFormLayout
import numpy as np
#Add the rest of the hyper paramaeters: create two panels, one next to another, and add them.

class DDPGConfig(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.policy = QLineEdit("MlpPolicy")
        self.policy.textChanged.connect(lambda val: self.set_param("policy", val))

        self.verbose = QLineEdit("0")
        self.verbose.textChanged.connect(lambda val: self.set_param("verbose", val))

        self.action_noise = QLineEdit("None")
        self.action_noise.textChanged.connect(lambda val: self.set_param("action_noise", val))

        self.tau = QLineEdit("0.001")
        self.tau.textChanged.connect(lambda val: self.set_param("tau", val))

        self.batch_size = QLineEdit("128")
        self.batch_size.textChanged.connect(lambda val: self.set_param("batch_size", val))

        self.buffer_size = QLineEdit("50000")
        self.buffer_size.textChanged.connect(lambda val: self.set_param("buffer_size", val))

        self.tensorboard_log = QLineEdit("None")
        self.tensorboard_log.textChanged.connect(lambda val: self.set_param("tensorboard_log", val))

        self.policy_kwargs = QLineEdit("None")
        self.policy_kwargs.textChanged.connect(lambda val: self.set_param("policy_kwargs", val))


        self.seed = QLineEdit("None")
        self.seed.textChanged.connect(lambda val: self.set_param("seed", val))

        self.init_setup_model = QLineEdit("True")
        self.init_setup_model.textChanged.connect(lambda val: self.set_param("init_setup_model", val))


        self.gamma = QLineEdit("0.99")
        self.gamma.textChanged.connect(lambda val: self.set_param("gamma", val))
        
        
        layout.addRow("Policy:", self.policy)
        
        layout.addRow("Gamma:", self.gamma)
        layout.addRow("Action Noise:", self.action_noise)
        layout.addRow("Tau:", self.tau)
        layout.addRow("Batch size:", self.batch_size)
        layout.addRow("Buffer size:", self.buffer_size)
        layout.addRow("Verbose:", self.verbose)
        layout.addRow("Tensorboard Log:", self.tensorboard_log)
        layout.addRow("Policy kwargs:", self.policy_kwargs)
        layout.addRow("Seed:", self.seed)
        layout.addRow("init_setup_model", self.init_setup_model)

        self.setLayout(layout)

        self.policy_value = self.policy.text()
        self.gamma_value = self.gamma.text()
        self.buffer_size_value = self.buffer_size.text()
        self.batch_size_value = self.batch_size.text()
        self.tau_value = self.tau.text()
        self.action_noise_value = self.action_noise.text()
        self.verbose_value = self.verbose.text()
        self.tensorboard_log_value = self.tensorboard_log.text()
        self.init_setup_model_value = self.init_setup_model.text()
        self.policy_kwargs_value = self.policy_kwargs.text()
        self.seed_value = self.seed.text()
        


    def set_param(self, name, value):
        setattr(self, f"{name}_value", value)

    # === Getters ===
    def get_policy_value(self):                         return self.policy_value
    def get_gamma_value(self):                          return float(self.gamma_value)
    def get_buffer_size_value(self):                    return int(self.buffer_size_value)
    def get_batch_size_value(self):                     return int(self.batch_size_value)
    def get_tau_value(self):                            return float(self.tau_value)
    def get_action_noise_value(self):                   return self.action_noise_value
    def get_verbose_value(self):                        return int(self.verbose_value)
    def get_tensorboard_log_value(self):                return self.tensorboard_log_value
    def get_init_setup_model_value(self):               return np.bool_(self.init_setup_model_value.strip().lower() == 'true')
    def get_policy_kwargs_value(self):                  return self.policy_kwargs_value
    def get_seed_value(self):                           return self.seed_value




    '''
    https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html
    
lass stable_baselines.ddpg.DDPG(policy, env, gamma=0.99, memory_policy=None, 
eval_env=None, nb_train_steps=50, nb_rollout_steps=100, nb_eval_steps=100, 
param_noise=None, action_noise=None, normalize_observations=False, tau=0.001, 
batch_size=128, param_noise_adaption_interval=50, normalize_returns=False, 
enable_popart=False, observation_range=(-5.0, 5.0), critic_l2_reg=0.0, 
return_range=(-inf, inf), actor_lr=0.0001, critic_lr=0.001, 
clip_norm=None, reward_scale=1.0, render=False, render_eval=False, 
memory_limit=None, buffer_size=50000, random_exploration=0.0, 
verbose=0, tensorboard_log=None, _init_setup_model=True, 
policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)
    '''