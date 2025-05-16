import numpy as np
import torch
from .bam_laplace import Laplace
from .bam_curvlinops import CurvlinopsGGN

from bam_torch.predicting.evaluator import Evaluator
from bam_torch.utils.utils import get_dataloader
from time import time

import matplotlib.pyplot as plt


class PostHocLaplace(Evaluator):
    def __init__(self, json_data):
        super().__init__(json_data)
    
    def get_train_loader(self):
        json_data = self.json_data
        train_loader, valid_loader, uniq_element, enr_avg_per_element = \
                                                        get_dataloader(
                                                            json_data['fname_traj'],
                                                            json_data['ntrain'],
                                                            json_data['nvalid'],
                                                            json_data['nbatch'],
                                                            json_data['cutoff'],
                                                            json_data['NN']['data_seed'],
                                                            json_data['element'],
                                                            json_data['regress_forces'],
                                                            self.rank,
                                                            self.world_size
                                                        )
        return train_loader, valid_loader, uniq_element, enr_avg_per_element
    
    def laplace_approximate(self, dict_key_y='energy'):
        """
        dict_key_y: 'energy' or 'forces_x' or 'forces_y' or 'forces_z' 
            A key for outputs of model
        """
        train_loader, _, _, _ = self.get_train_loader()
        e_corr = torch.tensor(self.model_ckpt['valid_scale_shift']) # or train_scale_shift
        e_corr = e_corr.flatten().mean()
        scale_info = {
            'e_corr': e_corr,
            'enr_avg_per_element': self.enr_avg_per_element
        }
        stochastic = False         # True: Fischer MC integral with 'num_samples'
                                  # False: GGN (default)
        num_samples = 10          # default=1
        backend_kwargs = {
            'scale_info': scale_info,
            'stochastic': stochastic,
            'num_samples': num_samples
        }
        
        la = Laplace(
            model=self.model, 
            likelihood="regression",  # 'regression' or 'classification'
            subset_of_weights='all',  # 'all' or 'subnetwork' or 'last_layer'
            hessian_structure='diag', # 'diag' or 'kron' (110.25 GB) or 'full' (5079.35 GB)
            dict_key_y=dict_key_y,
            backend=CurvlinopsGGN,    # default
            backend_kwargs=backend_kwargs
        )
        la.fit(train_loader)
        log_prior, log_sigma = (
            torch.ones(1, requires_grad=True),
            torch.ones(1, requires_grad=True),
        )
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
        #nepoch = self.json_data['NN']['nepoch']
        print('start_epoch:', self.start_epoch)
        t1 = time()
        for i in range(self.start_epoch):
            hyper_optimizer.zero_grad()
            neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
            neg_marglik.backward()
            hyper_optimizer.step()
        print(f"===+--> Elapsed time of hyperparams tuning: {time()-t1}")
        state_dict = la.state_dict()
        torch.save(state_dict, "state_dict.bin")
        la.load_state_dict(torch.load("state_dict.bin"))

        print(
            f"sigma={la.sigma_noise.item():.2f}",
            f"prior precision={la.prior_precision.item():.2f}",
        )

        #x = X_test.flatten().cpu().numpy()
        data_points = []
        y_train = []
        #x = [] # test 
        f_mu_list = []
        pred_std_list = []
        elapsed_time = []
        for i, data in enumerate(self.data_loader):
            data_points.append(i)
            y_train.append(data['energy'])

            t1 = time()
            data.to(self.device)
            f_mu, f_var = la(data)
            f_mu_joint, f_cov = la(data, joint=True)
            print(f"\n =======+------ {i}-th Data -------+=======")
            print(f"f_mu = {f_mu}")
            print(f"f_var = {f_var}")        
            assert torch.allclose(f_mu.flatten(), f_mu_joint)
            assert torch.allclose(f_var.flatten(), f_cov.diag())

            f_mu = f_mu.squeeze().detach().cpu().numpy()
            f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
            pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)
            f_mu_list.append(f_mu.item())
            pred_std_list.append(pred_std)
            elapsed_time.append(time()-t1)
            print(f"f_mu_joint = {f_mu_joint}")
            print(f"f_cov = {f_cov}")

            print(f"==+-> f_mu = {f_mu}")
            print(f"==+-> f_sigma = {f_sigma}")
            print(f"==+-> pred_std = {pred_std}\n")
            print(f"===+--> Elapsed time: {time()-t1}")
        print(f"\n====+---> Total Elapsed time: {sum(elapsed_time)}\n")
        plot_regression(
            torch.tensor(data_points), 
            torch.tensor(y_train), 
            torch.tensor(data_points),
            torch.tensor(f_mu_list), 
            torch.tensor(pred_std_list), 
            file_name="regression_example-energy", 
            plot=True
        )


def plot_regression(
    X_train, y_train, X_test, f_test, y_std, plot=True, file_name="regression_example"
):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(4.5, 2.8))
    ax1.set_title("MAP")
    ax1.scatter(X_train.flatten(), y_train.flatten(), alpha=0.3, color="tab:orange")
    ax1.plot(X_test, f_test, color="black", label=r"$f_{MAP}$")
    ax1.legend()

    ax2.set_title("LA")
    ax2.scatter(X_train.flatten(), y_train.flatten(), alpha=0.3, color="tab:orange")
    ax2.plot(X_test, f_test, label=r"$\mathbb{E}[f]$")
    ax2.fill_between(
        X_test,
        f_test - y_std * 2,
        f_test + y_std * 2,
        alpha=0.3,
        color="tab:blue",
        label=r"$2\sqrt{\mathbb{V}\,[y]}$",
    )
    ax2.legend()
    #ax1.set_ylim([-4, 6])
    ax1.set_xlim([X_test.min(), X_test.max()])
    ax2.set_xlim([X_test.min(), X_test.max()])
    ax1.set_ylabel("$y$")
    ax1.set_xlabel("$x$")
    ax2.set_xlabel("$x$")
    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig(f"{file_name}.png")


from bam_torch.utils.utils import find_input_json, date
import json
import torch

if __name__ == '__main__':
    print(date()) 
    input_json_path = find_input_json()
    torch.cuda.empty_cache()

    with open(input_json_path) as f:
        json_data = json.load(f)

        approximator = PostHocLaplace(json_data)
        dict_key_y = 'energy' # 'energy' or 'forces_x' or 'forces_y' or 'forces_z' 
        approximator.laplace_approximate(dict_key_y)

    print(date())