from .utils import date
from copy import deepcopy

class Logger:
    def __init__(self, log_config, loss_config=None, log_length='simple'):
        """ Default of log_config
        log_config = {'step': ['date', 'epoch'],
                      'train': ['loss', 'loss_e', 'loss_f'],
                      'valid': ['loss', 'loss_e', 'loss_f'],
                      'lr': ['lr']}
        loss_config = {'energy_loss': 'mse', 'force_loss': 'mse'}
        """
        self.log_config = log_config
        self.loss_config = loss_config
        self.length = 5
        self.space = 11
        if log_length == 'precise':
            self.length = 7
            self.space = 13
        self.logger_config = self.configure_logger_head()

    def configure_logger_head(self):
        if self.loss_config != None:
            log_config = deepcopy(self.log_config)
            for key, values in self.log_config.items():
                for i, value in enumerate(values):
                    if value == 'loss_e':
                        log_config[key][i] = f"{self.loss_config['energy_loss']}_e"
                    elif value == 'loss_f':
                        log_config[key][i] = f"{self.loss_config['force_loss']}_f"
        divider = " | "
        logger = {}
        keys = list(log_config.keys())
        for k in range(len(log_config)):
            if k == 0:
                key = keys[k].upper()
                line = f"{'MM/DD/YYYY HH/MM/SS':<23}"
                space = [23]
                key_values = list(log_config.values())[k]
                del key_values[0]
                for v in range(len(key_values)):
                    line += f"{key_values[v].upper():{len(key_values[v])+2}}"
                    space += [len(key_values[v])+2]
                line += divider
                logger[f'{key}'] = [line, space]
            else:
                key = keys[k].upper()
                line = ""
                space = []
                key_values = list(log_config.values())[k]
                if k < 3: #len(log_config)-1:
                    spc = self.space
                else:
                    spc = self.space-6
                for v in range(len(key_values)):
                    line += f"{key_values[v].upper():{spc}}"
                    space += [spc]
                if k < 3: #len(log_config)-1:
                    line += divider
                logger[f'{key}'] = [line, space]
                
        return logger

    def print_logger_head(self, fout):
        head = ""
        LINE = ""
        keys = list(self.logger_config.keys())
        values = list(self.logger_config.values())
        divider = "| "
        for k in range(len(self.logger_config)):
            if k == 0:
                key_values = values[k]
                line = key_values[0]
                separator = ' ' * (len(line)-len(divider))
                head += separator
                LINE += line
            elif k < 3: #len(self.logger_config)-1:
                key_values = values[k]
                key = keys[k]
                line = key_values[0]
                separator = '_' * (len(line)-len(divider)-len(key))
                head = head + divider + key + separator
                LINE += line
            else:
                key_values = values[k]
                key = keys[k]
                line = key_values[0]
                head += divider
                LINE += line
        separator = '-' * len(LINE)
        print(head, file=fout)
        print(LINE, file=fout)
        print(separator, file=fout)
        fout.flush()
        print(date())
        print(head)
        print(LINE)
        print(separator)

        return separator
    
    def print_epoch_loss(self, step_dict, epoch_loss_train, epoch_loss_valid, lr=None, fout=None):

        keys = list(self.log_config.keys())
        assert len(step_dict) == len(self.log_config[keys[0]])         # step
        assert len(epoch_loss_train) == len(self.log_config[keys[1]])  # train or predict
        assert len(epoch_loss_valid) == len(self.log_config[keys[2]])  # valid or exact
        
        values = list(self.log_config.values())
        _intervals = list(self.logger_config.values())
        intervals = [item[1] for item in _intervals]

        LINE = ""
        divider = " | "
        key = values[0]
        for k in range(len(key)):
            LINE += f"{step_dict[key[k]]:<{intervals[0][k]}}"
        LINE += divider

        key = values[1]
        for k in range(len(key)):
            LINE += f"{float(epoch_loss_train[key[k]]):<{intervals[1][k]}.{self.length}g}"
        LINE += divider

        key = values[2]
        for k in range(len(key)):
            LINE += f"{float(epoch_loss_valid[key[k]]):<{intervals[2][k]}.{self.length}g}"
        
        if lr != None:
            LINE += divider
            LINE += f"{lr:<{intervals[3][0]}.4g}"
        
        print(LINE, file=fout)
        fout.flush()
        print(LINE)

