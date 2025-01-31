from base_trainer import BaseTrainer

class MeanVarianceEstimationTrainer(BaseTrainer):
    """Trainer for mean-variance estimation (MVE) model
    """
    def __init__(self, json_data):
        super().__init__()
                
    def train(self):
        # TODO: Implement this method 
        raise NotImplementedError("This method needs to be implemented.")
    
    def configure_logger_head(self):
        self.log_config = self.json_data.get("log_config")
        if self.log_config == None:
            if json_data["regress_forces"]:
                """ You are free to add the values you want to include in the log file.
                The ```Logger``` will automatically generate the log file format, 
                and you have to insert values into the ```Logger``` 
                according to the defined default ```log_config``` here.

                Or simply sepecify in the "input.json"
                
                You can refer to the example provided in ```BaseTrainer```.
                """
                self.log_config = {
                    'step': ['date', 'epoch'], # If DeepEnsemble, you can insert 'istate' 
                    'train': ['loss', 'loss_e', 'loss_f', 'loss_l2'], # In this case, you must set l2_lambda
                    'valid': ['loss', 'loss_e', 'loss_f'],
                    'lr': ['lr'],
                    }  # loss_l2
            else:
                self.log_config = {
                    'step': ['date', 'epoch'],
                    'train': ['loss', 'loss_e', 'enr_var'], # In this case, you must calculate enr_var
                    'valid': ['loss', 'loss_e'],
                    'lr': ['lr'],
                    }
        # TODO: Implement this method 
        # Set default log configuration
        # or please sepecify in the "input.json"
        raise NotImplementedError("This method needs to be implemented.")
