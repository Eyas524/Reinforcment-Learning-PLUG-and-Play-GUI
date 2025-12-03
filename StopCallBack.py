from stable_baselines3.common.callbacks import BaseCallback

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, gui_instance, verbose=0):
        super().__init__(verbose)
        # Store the GUI instance where stop_training flag lives
        self.gui_instance = gui_instance

    def _on_step(self) -> bool:
        # Check flag from GUI
        if getattr(self.gui_instance, "stop_training", False):
            print("Manual stop triggered")
            return False  # stops training cleanly
        return True