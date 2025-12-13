"""
Multihead Evaluator for BAM-torch / RACEUnified models.

Extends the original BAM-torch Evaluator to support:
- Multihead models (RACEUnified with multiple heads)
- Single-head models (original RACE)
- Auto-detection of model type from checkpoint

Usage:
    python evaluate.py
    
Configuration in input.json:
    "predict": {
        "model": "path/to/model.pkl",
        "eval_head": 0,  // Head index to evaluate (for multihead models)
        ...
    }
"""

import torch
import gc
import json
import os
from e3nn import o3
from bam_torch.predicting.evaluator import Evaluator
from bam_torch.utils.utils import find_input_json, date
from bam_torch.model.models import RACEUnified


class MultiheadEvaluator(Evaluator):
    """
    Evaluator that supports both single-head and multihead models.
    
    Auto-detects model type from checkpoint:
    - If checkpoint has 'multihead.datasets' -> multihead mode
    - Otherwise -> single-head mode
    """
    
    def __init__(self, json_data, rank=0, world_size=1):
        # Apply predict.loss_config to NN.loss_config BEFORE parent init
        pd_config = json_data.get("predict", {})
        if pd_config.get("loss_config") is not None:
            json_data["NN"]["loss_config"] = pd_config.get("loss_config")
            print(f"✓ Using predict.loss_config: {pd_config.get('loss_config')}")
        
        super().__init__(json_data, rank, world_size)
    
    def configure_model(self):
        """Override: Configure model to support RACEUnified with multihead."""
        
        # Get model path from predict config
        predict_config = self.json_data.get('predict', {})
        model_path = predict_config.get('model', self.json_data['NN'].get('fname_pkl'))
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Check if multihead model
        ckpt_input_json = ckpt.get('input.json', {})
        multihead_config = ckpt_input_json.get('multihead', {})
        datasets = multihead_config.get('datasets', [])
        
        if datasets and len(datasets) > 0:
            # Multihead model
            heads = [ds.get('name', f'head_{i}') for i, ds in enumerate(datasets)]
            print(f"✅ Loading RACEUnified with {len(heads)} heads: {heads}")
        else:
            # Single-head model
            heads = None
            print(f"✅ Loading RACEUnified (single-head mode)")
        
        # Get model config from checkpoint or current json_data
        model_config = ckpt_input_json if ckpt_input_json else self.json_data
        
        hidden_irreps = o3.Irreps(model_config.get('hidden_channels', self.json_data['hidden_channels']))
        features_dim = model_config.get('features_dim', self.json_data['features_dim'])
        nlayers = model_config.get('nlayers', self.json_data['nlayers'])
        cutoff = model_config.get('cutoff', self.json_data['cutoff'])
        num_basis_func = model_config.get('num_radial_basis', self.json_data['num_radial_basis'])
        max_ell = model_config.get('max_ell', self.json_data['max_ell'])
        avg_num_neighbors = model_config.get('avg_num_neighbors', self.json_data['avg_num_neighbors'])
        num_species = model_config.get('num_species', self.json_data['num_species'])
        
        output_irreps = model_config.get('output_channels', "1x0e")
        active_fn = model_config.get('active_fn', "identity")
        
        regress_forces = model_config.get('regress_forces', self.json_data.get('regress_forces'))
        if regress_forces == True:
            regress_forces = "autograd"
        elif regress_forces == False:
            regress_forces = "false"
        
        mlp_irreps = o3.Irreps(f"{features_dim}x0e")
        
        # Create RACEUnified model
        model = RACEUnified(
            cutoff=cutoff,
            avg_num_neighbors=avg_num_neighbors,
            num_species=num_species,
            max_ell=max_ell,
            num_basis_func=num_basis_func,
            hidden_irreps=hidden_irreps,
            nlayers=nlayers,
            features_dim=features_dim,
            output_irreps=output_irreps,
            active_fn=active_fn,
            radial_MLP=[64, 64],
            MLP_irreps=mlp_irreps,
            regress_forces=regress_forces,
            compute_stress=True,
            heads=heads,
            cueq_config=None,
        )
        
        # Load weights
        state_dict = ckpt['params']
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        start_epoch = 0
        
        print(f"  - Total parameters: {n_params:,}")
        
        return model, n_params, ckpt, start_epoch
    
    def compute_loss(self, preds, data):
        """Override: Handle missing stress data gracefully."""
        lambda_config = self.json_data.get("NN", {})
        e_lambda = lambda_config.get('enr_lambda', 1.0)
        f_lambda = lambda_config.get('frc_lambda', 1.0)
        s_lambda = lambda_config.get('str_lambda', 1.0)
        
        loss = {"loss": []}
        
        # Energy loss
        energy_target = data["energy"].flatten()
        if self.loss_fn.get("energy_loss") is not None:
            loss["loss_e"] = self.loss_fn["energy_loss"](preds["energy"].flatten(), energy_target)
            loss["loss"].append(e_lambda * loss["loss_e"])
        
        # Force loss
        if "forces" in preds and self.loss_fn.get('force_loss') is not None:
            force_target = data["forces"].flatten()
            loss["loss_f"] = self.loss_fn["force_loss"](preds["forces"].flatten(), force_target)
            loss["loss"].append(f_lambda * loss["loss_f"])
        
        # Stress loss (only if data has stress)
        if "stress" in preds and self.loss_fn.get('stress_loss') is not None:
            try:
                stress_target = data["stress"].flatten()
                loss["loss_s"] = self.loss_fn["stress_loss"](preds["stress"].flatten(), stress_target)
                loss["loss"].append(s_lambda * loss["loss_s"])
            except (KeyError, AttributeError):
                pass  # No stress in data
        
        loss["loss"] = sum(loss["loss"]) if loss["loss"] else torch.tensor(0.0)
        return loss
    
    def configure_logger(self):
        """Override: Configure logger without on_exit callback to avoid file errors."""
        from bam_torch.utils.logger import Logger
        import atexit
        
        log_config = {
            'step': ['date', 'data'],
            'predict': ['energy', 'loss_e', 'loss_f'],
            'exact': ['energy']
        }
        
        log_length = self.json_data.get("plog_length", "precise")
        log_interval = 1
        
        predict_config = self.json_data.get('predict', {})
        fname = predict_config.get('fname_plog', "predict.out")
        
        self.fout = open(fname, 'w')
        logger = Logger(log_config, self.loss_config, log_length, self.fout)
        
        # Simple cleanup - just close the file, don't call on_exit
        atexit.register(lambda: self.fout.close() if not self.fout.closed else None)
        
        return log_config, log_interval, logger, self.fout
    
    def evaluate(self, element_wise=True):
        """
        BAM-torch style evaluate with multihead support.
        """
        self.logger.print_logger_head()
        
        target = {}
        eval_loss_dict = {'loss': [], 'loss_e': [], 'loss_f': []}
        
        #  Check if Multihead or not (from input.json saved in model.pkl)
        ckpt_input_json = self.model_ckpt.get('input.json', {})
        multihead_config = ckpt_input_json.get('multihead', {})
        datasets = multihead_config.get('datasets', [])
        is_multihead = len(datasets) > 0
        
        if is_multihead:
            num_heads = len(datasets)
            head_names = [ds.get('name', f'head_{i}') for i, ds in enumerate(datasets)]
            
            # Select the head from input.json (Default: 0 = target head)
            eval_head = self.json_data.get('predict', {}).get('eval_head', 0)
            if isinstance(eval_head, str):
                eval_head = head_names.index(eval_head) if eval_head in head_names else 0
            
            print(f"\n✅ Multihead model detected: {num_heads} heads")
            print(f"   Evaluating with head {eval_head}: {head_names[eval_head]}\n")
            
            # ⭐ Use per-head E0s if available
            per_head_enr_avg = self.model_ckpt.get('per_head_enr_avg', {})
            if eval_head in per_head_enr_avg:
                head_e0s = per_head_enr_avg[eval_head].get('enr_avg_per_element', self.enr_avg_per_element)
                print(f"   ✓ Using per-head E0s for head {eval_head}")
            else:
                head_e0s = self.enr_avg_per_element
                print(f"   ⚠️ No per-head E0s found, using global E0s")
        else:
            print(f"\n✅ Single-head model detected\n")
            head_e0s = self.enr_avg_per_element
        
        # Get e_corr
        # For multihead, try to use per_head_scale_shift
        if is_multihead:
            per_head_scale_shift = self.model_ckpt.get('per_head_scale_shift', {})
            if eval_head in per_head_scale_shift and per_head_scale_shift[eval_head]:
                head_shifts = per_head_scale_shift[eval_head]
                e_corr_ = sum(head_shifts) / len(head_shifts)
                element_wise = False
                print(f"   ✓ Using per-head scale_shift: {e_corr_:.4f}")
            else:
                e_corr_ = torch.tensor(0.0)
                element_wise = False
                print(f"   ⚠️ No per-head scale_shift found, using 0")
        else:
            e_corr_, element_wise = self.get_scale_shift_correction(element_wise)    

        test_values = {
            'energy': [],
            'force_x': [],
            'force_y': [],
            'force_z': [],
            'exact_energy': [],
            'exact_force_x': [],
            'exact_force_y': [],
            'exact_force_z': [],
        }
        for i, data in enumerate(self.data_loader):
            data = data.to(self.device)
            
            # Get node_enr_avg using per-head E0s
            species = data['species']
            node_enr_avg = torch.tensor(
                [head_e0s.get(int(iz), 0.0) for iz in species],
            ).sum()
            
            # Multihead: head setting
            if is_multihead:
                num_graphs = data['batch'].max().item() + 1
                data['head'] = torch.full((num_graphs,), eval_head, dtype=torch.long, device=self.device)
            
            # Predict energy, forces, and so on
            preds = self.model(data, backprop=False)
            
            # Correct the energy
            if element_wise:
                e_corr = torch.tensor(
                    [e_corr_[int(iz)] for iz in species]
                ).sum()
            else:
                e_corr = e_corr_
            preds['energy'] = preds["energy"] + node_enr_avg + e_corr

            test_values['energy'].append(preds['energy'].detach().cpu())
            test_values['force_x'].append(preds['forces'][:,0].detach().cpu())
            test_values['force_y'].append(preds['forces'][:,1].detach().cpu())
            test_values['force_z'].append(preds['forces'][:,2].detach().cpu())
            test_values['exact_energy'].append(data['energy'].detach().cpu())
            test_values['exact_force_x'].append(data['forces'][:,0].detach().cpu())
            test_values['exact_force_y'].append(data['forces'][:,1].detach().cpu())
            test_values['exact_force_z'].append(data['forces'][:,2].detach().cpu())

            # loss computation
            loss_dict = self.compute_loss(preds, data)
            
            for l in eval_loss_dict.keys():
                eval_loss_dict[l].append(loss_dict.get(l, torch.nan).detach().cpu())
            
            # Print evaluation loss
            pred_energy = float(preds['energy'][0].detach().cpu())
            exact_energy = float(data['energy'][0].detach().cpu())
            mae_e = float(loss_dict.get('loss_e', 0.0).detach().cpu()) if isinstance(loss_dict.get('loss_e'), torch.Tensor) else 0.0
            mae_f = float(loss_dict.get('loss_f', 0.0).detach().cpu()) if isinstance(loss_dict.get('loss_f'), torch.Tensor) else 0.0
            
            print(f"{date()}  {i:6d} | {pred_energy:<15.6f} {mae_e:<15.6g} {mae_f:<15.6g} | {exact_energy:<15.6f} |")
            print(f"{date()}  {i:6d} | {pred_energy:<15.6f} {mae_e:<15.6g} {mae_f:<15.6g} | {exact_energy:<15.6f} |", file=self.fout)
            
            # Free memory
            del data, preds, loss_dict
            torch.cuda.empty_cache()
            if i % 100 == 0:
                gc.collect()
        
        # Get means of loss values per epoch
        eval_loss_dict = {key: torch.mean(torch.tensor(value)) 
                          for key, value in eval_loss_dict.items()}
        
        separator = self.logger.get_seperator()
        print(separator, file=self.fout)
        print(separator)
        
        if is_multihead:
            print(f"HEAD: {head_names[eval_head]} (index: {eval_head})", file=self.fout)
            print(f"HEAD: {head_names[eval_head]} (index: {eval_head})")
        
        print(f"MEAN_LOSS(E): {eval_loss_dict['loss_e']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS(F): {eval_loss_dict['loss_f']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS(E): {eval_loss_dict['loss_e']:<11.5g}")
        print(f"MEAN_LOSS(F): {eval_loss_dict['loss_f']:<11.5g}\n")
        torch.save(test_values, "test_values.pkl")
        

if __name__ == '__main__':
    print(date())
    input_json_path = find_input_json()
    torch.cuda.empty_cache()

    with open(input_json_path) as f:
        json_data = json.load(f)

    if json_data.get('predict'):
        evaluator = MultiheadEvaluator(json_data)
        evaluator.evaluate()
    else:
        print('No predict config found in input.json')

    print(date())
