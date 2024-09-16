from datetime import datetime
import os

def create_log_dirs(base_dir):
    # Create log saving global model_attack
    log_dir = os.path.join(base_dir, "log_global_model")
    os.makedirs(log_dir, exist_ok=True)

    # Create log saving tensorboard
    log_tensorboard = os.path.join(base_dir, "log_tensorboard_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_tensorboard, exist_ok=True)

    # Create log saving client model_attack
    log_client = os.path.join(base_dir, "log_client")
    os.makedirs(log_client, exist_ok=True)

    return log_dir, log_tensorboard, log_client