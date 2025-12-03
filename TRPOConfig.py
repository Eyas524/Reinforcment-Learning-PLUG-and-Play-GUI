from PyQt5.QtWidgets import QWidget, QLineEdit, QFormLayout, QLabel
import numpy as np
class TRPOConfig(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.policy = QLineEdit("MlpPolicy")
        self.policy.textChanged.connect(lambda val: self.set_param("policy", val))

        self.learning_rate = QLineEdit("0.0003")
        self.learning_rate.textChanged.connect(lambda val: self.set_param("learning_rate", val))

        self.n_steps_per_batch = QLineEdit("1024")  # timesteps_per_batch
        self.n_steps_per_batch.textChanged.connect(lambda val: self.set_param("n_steps_per_batch", val))

        self.gamma = QLineEdit("0.99")
        self.gamma.textChanged.connect(lambda val: self.set_param("gamma", val))

        self.max_kl = QLineEdit("0.01")
        self.max_kl.textChanged.connect(lambda val: self.set_param("max_kl", val))

        self.cg_iters = QLineEdit("10")
        self.cg_iters.textChanged.connect(lambda val: self.set_param("cg_iters", val))

        self.lam = QLineEdit("0.98")
        self.lam.textChanged.connect(lambda val: self.set_param("lam", val))

        self.entcoeff = QLineEdit("0.0")
        self.entcoeff.textChanged.connect(lambda val: self.set_param("entcoeff", val))

        self.cg_damping = QLineEdit("0.01")
        self.cg_damping.textChanged.connect(lambda val: self.set_param("cg_damping", val))

        self.vf_stepsize = QLineEdit("0.0003")
        self.vf_stepsize.textChanged.connect(lambda val: self.set_param("vf_stepsize", val))

        self.vf_iters = QLineEdit("3")
        self.vf_iters.textChanged.connect(lambda val: self.set_param("vf_iters", val))

        self.verbose = QLineEdit("0")
        self.verbose.textChanged.connect(lambda val: self.set_param("verbose", val))

        self.tensorboard_log = QLineEdit("None")
        self.tensorboard_log.textChanged.connect(lambda val: self.set_param("tensorboard_log", val))

        self.policy_kwargs = QLineEdit("None")
        self.policy_kwargs.textChanged.connect(lambda val: self.set_param("policy_kwargs", val))

        self.full_tensorboard_log = QLineEdit("False")
        self.full_tensorboard_log.textChanged.connect(lambda val: self.set_param("full_tensorboard_log", val))

        self.init_setup_model = QLineEdit("True")
        self.init_setup_model.textChanged.connect(lambda val: self.set_param("init_setup_model", val))

        self.seed = QLineEdit("None")
        self.seed.textChanged.connect(lambda val: self.set_param("seed", val))

        self.n_cpu_tf_sess = QLineEdit("1")
        self.n_cpu_tf_sess.textChanged.connect(lambda val: self.set_param("n_cpu_tf_sess", val))

        #self.link_info = QLineEdit("https://stable-baselines.readthedocs.io/en/master/modules/trpo.html")

        # === Add to layout ===
        layout.addRow("Policy:", self.policy)
        #layout.addRow("Learning Rate:", self.learning_rate) #not required
        layout.addRow("Gamma:", self.gamma)
        #layout.addRow("Timesteps per batch:", self.n_steps_per_batch)
        #layout.addRow("Max KL:", self.max_kl)
        #layout.addRow("CG Iters:", self.cg_iters)
        #layout.addRow("Lambda (GAE):", self.lam)
        #layout.addRow("Entropy Coef:", self.entcoeff)
        layout.addRow("CG Damping:", self.cg_damping)
        #layout.addRow("VF Step Size:", self.vf_stepsize)
        #layout.addRow("VF Iters:", self.vf_iters)
        layout.addRow("Verbose:", self.verbose)
        #layout.addRow("Tensorboard Log:", self.tensorboard_log)
        layout.addRow("Init Setup Model", self.init_setup_model)
        #layout.addRow("Policy kwargs:", self.policy_kwargs)
        #layout.addRow("Full Tensorboard Log:", self.full_tensorboard_log)
        layout.addRow("Seed:", self.seed)
        #layout.addRow("Num CPU TF Sess:", self.n_cpu_tf_sess)
        #layout.addRow("Link", self.link_info)

        self.setLayout(layout)

        # === Store initial values ===
        self.policy_value = self.policy.text()
        self.learning_rate_value = self.learning_rate.text()
        self.n_steps_per_batch_value = self.n_steps_per_batch.text()
        self.gamma_value = self.gamma.text()
        self.max_kl_value = self.max_kl.text()
        self.cg_iters_value = self.cg_iters.text()
        self.lam_value = self.lam.text()
        self.entcoeff_value = self.entcoeff.text()
        self.cg_damping_value = self.cg_damping.text()
        self.vf_stepsize_value = self.vf_stepsize.text()
        self.vf_iters_value = self.vf_iters.text()
        self.verbose_value = self.verbose.text()
        self.tensorboard_log_value = self.tensorboard_log.text()
        self.init_setup_model_value = self.init_setup_model.text()
        self.policy_kwargs_value = self.policy_kwargs.text()
        self.full_tensorboard_log_value = self.full_tensorboard_log.text()
        self.seed_value = self.seed.text()
        self.n_cpu_tf_sess_value = self.n_cpu_tf_sess.text()

    def set_param(self, name, value):
        setattr(self, f"{name}_value", value)

    # === Getters ===
    def get_policy_value(self): return self.policy_value
    def get_learning_rate_value(self): return self.learning_rate_value# not required
    def get_gamma_value(self): return self.gamma_value
    def get_timesteps_per_batch_value(self): return self.n_steps_per_batch_value
    def get_max_kl_value(self): return self.max_kl_value
    def get_cg_iters_value(self): return self.cg_iters_value
    def get_lam_value(self): return self.lam_value
    def get_entcoeff_value(self): return self.entcoeff_value
    def get_cg_damping_value(self): return self.cg_damping_value
    def get_vf_stepsize_value(self): return self.vf_stepsize_value
    def get_vf_iters_value(self): return self.vf_iters_value
    def get_verbose_value(self): return int(self.verbose_value)
    def get_tensorboard_log_value(self): return self.tensorboard_log_value
    def get_init_setup_model_value(self): return np.bool_(self.init_setup_model_value.strip().lower() == 'true') #return np.bool_(self.init_setup_model_value.strip().lower() == 'true')
    def get_policy_kwargs_value(self): return self.policy_kwargs_value
    def get_full_tensorboard_log_value(self): return self.full_tensorboard_log_value
    def get_seed_value(self): return self.seed_value
    def get_n_cpu_tf_sess_value(self): return self.n_cpu_tf_sess_value


'''
https://stable-baselines.readthedocs.io/en/master/modules/trpo.html


class stable_baselines.trpo_mpi.TRPO(policy, env, 
gamma=0.99, timesteps_per_batch=1024, max_kl=0.01,
 cg_iters=10, lam=0.98, entcoeff=0.0, cg_damping=0.01, 
 vf_stepsize=0.0003, vf_iters=3, verbose=0, tensorboard_log=None, 
_init_setup_model=True, policy_kwargs=None,
 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)[source]
'''