from PyQt5.QtWidgets import QWidget, QLineEdit, QFormLayout

class PPOConfig(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.policy = QLineEdit("MlpPolicy")
        self.policy.textChanged.connect(lambda val: self.set_param("policy", val))

        self.verbose = QLineEdit("1")
        self.verbose.textChanged.connect(lambda val: self.set_param("verbose", val))

        self.learning_rate = QLineEdit("0.0003")
        self.learning_rate.textChanged.connect(lambda val: self.set_param("learning_rate", val))

        self.n_steps = QLineEdit("2048")
        self.n_steps.textChanged.connect(lambda val: self.set_param("n_steps", val))
        
        self.batch_size = QLineEdit("64")
        self.batch_size.textChanged.connect(lambda val: self.set_param("batch_size", val))

        self.n_epochs = QLineEdit("10")
        self.n_epochs.textChanged.connect(lambda val: self.set_param("n_epochs", val))
        
        self.gamma = QLineEdit("0.99")
        self.gamma.textChanged.connect(lambda val: self.set_param("gamma", val))

        self.clip_range = QLineEdit("0.2")
        self.clip_range.textChanged.connect(lambda val: self.set_param("clip_range", val))

        self.clip_range_vf = QLineEdit("None")
        self.clip_range_vf.textChanged.connect(lambda val: self.set_param("clip_range_vf", val))

        self.normalize_advantage = QLineEdit("True")
        self.normalize_advantage.textChanged.connect(lambda val: self.set_param("normalize_advantage", val))
        
        self.ent_coef = QLineEdit("0.0")
        self.ent_coef.textChanged.connect(lambda val: self.set_param("ent_coef", val))
        
        self.vf_coef = QLineEdit("0.5")
        self.vf_coef.textChanged.connect(lambda val: self.set_param("vf_coef", val))
        
        self.max_grad_norm = QLineEdit("0.5")
        self.max_grad_norm.textChanged.connect(lambda val: self.set_param("max_grad_norm", val))
        
        self.gae_lambda = QLineEdit("0.95")
        self.gae_lambda.textChanged.connect(lambda val: self.set_param("gae_lambda", val))

        self.use_sde = QLineEdit("False")
        self.use_sde.textChanged.connect(lambda val: self.set_param("use_sde", val))

        self.sde_sample_freq = QLineEdit("-1")
        self.sde_sample_freq.textChanged.connect(lambda val: self.set_param("sde_sample_freq", val))

        self.rollout_buffer_class = QLineEdit("None")
        self.rollout_buffer_class.textChanged.connect(lambda val: self.set_param("rollout_buffer_class", val))

        self.rollout_buffer_kwargs = QLineEdit("None")
        self.rollout_buffer_kwargs.textChanged.connect(lambda val: self.set_param("rollout_buffer_kwargs", val))

        self.init_setup_model = QLineEdit("True")
        self.init_setup_model.textChanged.connect(lambda val: self.set_param("init_setup_model", val))


        self.target_kl = QLineEdit("None")
        self.target_kl.textChanged.connect(lambda val: self.set_param("target_kl", val))

        self.stats_window_size = QLineEdit("100")
        self.stats_window_size.textChanged.connect(lambda val: self.set_param("stats_window_size", val))

        self.tensorboard_log = QLineEdit("None")
        self.tensorboard_log.textChanged.connect(lambda val: self.set_param("tensorboard_log", val))

        self.policy_kwargs = QLineEdit("None")
        self.policy_kwargs.textChanged.connect(lambda val: self.set_param("policy_kwargs", val))

        self.seed = QLineEdit("None")
        self.seed.textChanged.connect(lambda val: self.set_param("seed", val))

        self.device = QLineEdit("auto")
        self.device.textChanged.connect(lambda val: self.set_param("device", val))

        #self.link_info = QLineEdit("https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html")

        # Add fields to layout
        layout.addRow("Policy:", self.policy)
        layout.addRow("Verbose:", self.verbose)
        layout.addRow("Learning Rate:", self.learning_rate)
        layout.addRow("n_steps:", self.n_steps)
        layout.addRow("Batch Size:", self.batch_size)
        layout.addRow("n_epochs:", self.n_epochs)
        layout.addRow("Gamma:", self.gamma)
        layout.addRow("Clip Range:", self.clip_range)
        layout.addRow("Clip Range VF:", self.clip_range_vf)
        layout.addRow("Normalize Advantage:", self.normalize_advantage)
        layout.addRow("Ent Coef:", self.ent_coef)
        layout.addRow("VF Coef:", self.vf_coef)
        layout.addRow("Max Grad Norm:", self.max_grad_norm)
        layout.addRow("GAE Lambda:", self.gae_lambda)
        layout.addRow("Use SDE:", self.use_sde)
        layout.addRow("SDE Sample Freq:", self.sde_sample_freq)
        layout.addRow("Rollout Buffer Class", self.rollout_buffer_class)
        layout.addRow("Rollout Buffer Kwargs", self.rollout_buffer_kwargs)
        layout.addRow("Init Setup Model", self.init_setup_model)
        layout.addRow("Target KL:", self.target_kl)
        layout.addRow("Stats Window Size:", self.stats_window_size)
        layout.addRow("Tensorboard Log:", self.tensorboard_log)
        layout.addRow("Policy kwargs:", self.policy_kwargs)
        layout.addRow("Seed:", self.seed)
        layout.addRow("Device:", self.device)
        #layout.addRow("Link:", self.link_info)

        self.setLayout(layout)

        # Store initial values
        self.policy_value = self.policy.text()
        self.verbose_value = self.verbose.text()
        self.learning_rate_value = self.learning_rate.text()
        self.n_steps_value = self.n_steps.text()
        self.batch_size_value = self.batch_size.text()
        self.n_epochs_value = self.n_epochs.text()
        self.gamma_value = self.gamma.text()
        self.clip_range_value = self.clip_range.text()
        self.clip_range_vf_value = self.clip_range_vf.text()
        self.normalize_advantage_value = self.normalize_advantage.text()
        self.ent_coef_value = self.ent_coef.text()
        self.vf_coef_value = self.vf_coef.text()
        self.max_grad_norm_value = self.max_grad_norm.text()
        self.gae_lambda_value = self.gae_lambda.text()
        self.use_sde_value = self.use_sde.text()
        self.sde_sample_freq_value = self.sde_sample_freq.text()
        self.rollout_buffer_class_value = self.rollout_buffer_class.text()
        self.rollout_buffer_kwargs_value = self.rollout_buffer_kwargs.text()
        self.init_setup_model_value = self.init_setup_model.text()
        self.target_kl_value = self.target_kl.text()
        self.stats_window_size_value = self.stats_window_size.text()
        self.tensorboard_log_value = self.tensorboard_log.text()
        self.policy_kwargs_value = self.policy_kwargs.text()
        self.seed_value = self.seed.text()
        self.device_value = self.device.text()

    def set_param(self, name, value):
        setattr(self, f"{name}_value", value)

    # === Getters ===
    def get_policy_value(self): return self.policy_value
    def get_verbose_value(self): return self.verbose_value
    def get_learning_rate_value(self): return self.learning_rate_value
    def get_n_steps_value(self): return self.n_steps_value
    def get_batch_size_value(self): return self.batch_size_value
    def get_n_epochs_value(self): return self.n_epochs_value
    def get_gamma_value(self): return self.gamma_value
    def get_clip_range_value(self): return self.clip_range_value
    def get_clip_range_vf_value(self): return self.clip_range_vf_value
    def get_normalize_advantage_value(self): return self.normalize_advantage_value
    def get_ent_coef_value(self): return self.ent_coef_value
    def get_vf_coef_value(self): return self.vf_coef_value
    def get_max_grad_norm_value(self): return self.max_grad_norm_value
    def get_gae_lambda_value(self): return self.gae_lambda_value
    def get_use_sde_value(self): return self.use_sde_value
    def get_rollout_buffer_class_value(self): return self.rollout_buffer_class_value
    def get_rollout_buffer_kwargs_value(self): return self.rollout_buffer_kwargs_value
    def get_init_setup_model_value(self): return self.init_setup_model_value
    def get_sde_sample_freq_value(self): return self.sde_sample_freq_value
    def get_target_kl_value(self): return self.target_kl_value
    def get_stats_window_size_value(self): return self.stats_window_size_value
    def get_tensorboard_log_value(self): return self.tensorboard_log_value
    def get_policy_kwargs_value(self): return self.policy_kwargs_value
    def get_seed_value(self): return self.seed_value
    def get_device_value(self): return self.device_value




'''
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

 class stable_baselines3.ppo.PPO(policy, env, learning_rate=0.0003, n_steps=2048, batch_size=64,
 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True,
 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, rollout_buffer_class=None,
 rollout_buffer_kwargs=None, target_kl=None, stats_window_size=100, tensorboard_log=None, policy_kwargs=None,
 verbose=0, seed=None, device='auto', _init_setup_model=True)
'''