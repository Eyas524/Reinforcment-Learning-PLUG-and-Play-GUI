from PyQt5.QtWidgets import QWidget, QLineEdit, QFormLayout
import numpy as np

class SACConfig(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.policy = QLineEdit("MlpPolicy")
        self.policy.textChanged.connect(lambda val: self.set_param("policy", val))

        self.gamma = QLineEdit("0.99")
        self.gamma.textChanged.connect(lambda val: self.set_param("gamma", val))

        self.learning_rate = QLineEdit("0.0003")
        self.learning_rate.textChanged.connect(lambda val: self.set_param("learning_rate", val))

        self.buffer_size = QLineEdit("50000")
        self.buffer_size.textChanged.connect(lambda val: self.set_param("buffer_size", val))

        self.learning_starts = QLineEdit("100")
        self.learning_starts.textChanged.connect(lambda val: self.set_param("learning_starts", val))
        
        
        self.train_freq = QLineEdit("1")
        self.train_freq.textChanged.connect(lambda val: self.set_param("train_freq", val))

        self.batch_size = QLineEdit("64")
        self.batch_size.textChanged.connect(lambda val: self.set_param("batch_size", val))

        self.tau = QLineEdit("0.005")
        self.tau.textChanged.connect(lambda val: self.set_param("tau", val))


        self.ent_coef = QLineEdit("auto")
        self.ent_coef.textChanged.connect(lambda val: self.set_param("ent_coef", val))

        self.target_update_interval = QLineEdit("1")
        self.target_update_interval.textChanged.connect(lambda val: self.set_param("target_update_interval", val))

        self.gradient_steps = QLineEdit("1")
        self.gradient_steps.textChanged.connect(lambda val: self.set_param("gradient_steps", val))

        self.target_entropy = QLineEdit("auto")
        self.target_entropy.textChanged.connect(lambda val: self.set_param("target_entropy", val))

        self.action_noise = QLineEdit("None")
        self.action_noise.textChanged.connect(lambda val: self.set_param("action_noise", val))


        self.verbose = QLineEdit("0")
        self.verbose.textChanged.connect(lambda val: self.set_param("verbose", val))

        self.tensorboard_log = QLineEdit("None")
        self.tensorboard_log.textChanged.connect(lambda val: self.set_param("tensorboard_log", val))

        self.init_setup_model = QLineEdit("True")
        self.init_setup_model.textChanged.connect(lambda val: self.set_param("init_setup_model", val))


        self.policy_kwargs = QLineEdit("None")
        self.policy_kwargs.textChanged.connect(lambda val: self.set_param("policy_kwargs", val))

        self.seed = QLineEdit("None")
        self.seed.textChanged.connect(lambda val: self.set_param("seed", val))



        #self.link_info = QLineEdit("https://stable-baselines.readthedocs.io/en/master/modules/sac.html")

  

        self.setLayout(layout)


        layout.addRow("Policy:", self.policy)
        layout.addRow("Gamma:", self.gamma)
        layout.addRow("Learning Rate:", self.learning_rate)
        
        layout.addRow("Buffer size:", self.buffer_size)
        layout.addRow("Learning Starts:", self.learning_starts)
        layout.addRow("Train frequency:", self.train_freq)
        layout.addRow("Batch size:", self.batch_size)
        layout.addRow("Tau:", self.tau)
        layout.addRow("Ent Coef:", self.ent_coef)
        layout.addRow("Target Update Interval:", self.target_update_interval)
        layout.addRow("Gradient steps:", self.gradient_steps)
        layout.addRow("Target Entropy:", self.target_entropy)
        layout.addRow("Action Noise", self.action_noise)
        
        layout.addRow("Verbose:", self.verbose)
        layout.addRow("Tensorboard Log:", self.tensorboard_log)
        layout.addRow("Init Setup Model", self.init_setup_model)
        layout.addRow("Policy kwargs:", self.policy_kwargs)
        
        layout.addRow("Seed:", self.seed)
        
        #layout.addRow("Link", self.link_info)

 


        self.policy_value                   = self.policy.text()
        self.learning_rate_value            = self.learning_rate.text()
        self.buffer_size_value              = self.buffer_size.text()
        self.batch_size_value               = self.batch_size.text()
        self.learning_starts_value          = self.learning_starts.text()
        self.tau_value                      = self.tau.text()
        self.gamma_value                    = self.gamma.text()
        self.train_freq_value               = self.train_freq.text()
        self.gradient_steps_value           = self.gradient_steps.text()
        self.ent_coef_value                 = self.ent_coef.text()
        self.target_update_interval_value   = self.target_update_interval.text()
        self.target_entropy_value           = self.target_entropy.text()
        self.action_noise_value             = self.action_noise.text()
        self.verbose_value                  = self.verbose.text()
        self.init_setup_model_value         = self.init_setup_model.text()
        self.tensorboard_log_value          = self.tensorboard_log.text()
        self.policy_kwargs_value            = self.policy_kwargs.text()
        
        self.seed_value = self.seed.text()


    def set_param(self, name, value):
        setattr(self, f"{name}_value", value)

    # === Getters ===
    def get_policy_value(self):                             return self.policy_value
    def get_learning_rate_value(self):                      return float(self.learning_rate_value)
    def get_buffer_size_value(self):                        return int(self.buffer_size_value)
    def get_batch_size_value(self):                         return int(self.batch_size_value)
    def get_learning_starts_value(self):                    return int(self.learning_starts_value)
    def get_tau_value(self):                                return float(self.tau_value)
    def get_gamma_value(self):                              return float(self.gamma_value)
    def get_train_freq_value(self):                         return int(self.train_freq_value)
    def get_gradient_steps_value(self):                     return int(self.gradient_steps_value)
    def get_action_noise_value(self):                       return self.action_noise_value
    def get_ent_coef_value(self):                           return self.ent_coef_value
    def get_target_update_interval_value(self):             return int(self.target_update_interval_value)
    def get_target_entropy_value(self):                     return self.target_entropy_value
    def get_verbose_value(self):                            return int(self.verbose_value)
    def get_init_setup_model_value(self):                   return np.bool_(self.init_setup_model_value.strip().lower() == 'true')
    def get_tensorboard_log_value(self):                    return self.tensorboard_log_value
    def get_policy_kwargs_value(self):                      return self.policy_kwargs_value
    def get_seed_value(self):                               return self.seed_value

    '''
    def get_full_tensorboard_log_value(self): return self.full_tensorboard_log_value
    def get_n_cpu_tf_sess_value(self): return self.n_cpu_tf_sess_value
    def get_random_exploration_value(self): return self.random_exploration_value
    '''

'''
https://stable-baselines.readthedocs.io/en/master/modules/sac.html

class stable_baselines.sac.SAC(policy, env, 
gamma=0.99, learning_rate=0.0003, buffer_size=50000, 
learning_starts=100, train_freq=1, batch_size=64, tau=0.005, 
ent_coef='auto', target_update_interval=1, gradient_steps=1, 
target_entropy='auto', action_noise=None, random_exploration=0.0, verbose=0, tensorboard_log=None, 
_init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)[source]
'''