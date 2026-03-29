import torch
import dgl
import dgl.function as fn
import numpy as np

class ChannelAllocationEnv:
    def __init__(self, max_epi_t, k, max_num_nodes, device):
        self.max_epi_t = max_epi_t
        self.k = k   
        self.max_num_nodes = max_num_nodes 
        self.device = device

    def step(self, action):
        reward, sol_colored,sol_zero, done = self._take_action(action)
        
        ob = self._build_ob()
        self.sol_colored = sol_colored
        self.sol_zero = sol_zero
        info = {
            "colored": self.sol_colored,
            "zero":    self.sol_zero
        }

        return ob, reward, done, info
    
    def _take_action(self, action):
        undecided = self.x == self.k+1
        self.x[undecided] = action[undecided]
        self.t += 1
        
        for i in range(1,self.k+1):
            x1 = (self.x == i)
            self.g = self.g.to(self.device)
            self.g.ndata['h'] = x1.float()
            self.g.update_all(
                fn.copy_u('h', 'm'), 
                fn.sum(msg='m', out='h')
                )
            x1_deg = self.g.ndata.pop('h')
            
            ## forgive clashing
            clashed = x1 & (x1_deg > 0)
            self.x[clashed] = self.k+1
            x1_deg[clashed] = 0
        
        # graph clean up
        still_undecided = (self.x == self.k+1)
        seen_per_channel = []
        for c in range(1,self.k+1):
            self.g.ndata['match'] = (self.x == c).float()
            self.g.update_all(fn.copy_u('match', 'm'),fn.sum('m', 'agg')) 
            seen_c = (self.g.ndata['agg'] > 0) 
            seen_per_channel.append(seen_c)
        
        seen_all = torch.stack(seen_per_channel, dim=1)
        has_all_channels = seen_all.all(dim=1)
        to_set_zero = still_undecided & has_all_channels
        self.x[to_set_zero] = 0

        # fill timeout with zeros
        still_undecided = (self.x == self.k+1)
        timeout = (self.t == self.max_epi_t)
        self.x[still_undecided & timeout] = 0

        done = self._check_done()
        self.epi_t[~done] += 1

        # compute reward and solution
        colored_now = ((self.x >= 1) & (self.x <= self.k)).float().sum(dim=0)
        colored_now = colored_now.unsqueeze(0).expand(self.g.batch_size, -1)
        zero_now = (self.x == 0).float().sum(dim=0)
        zero_now = zero_now.unsqueeze(0).expand(self.g.batch_size, -1)     
        
        delta_colored = colored_now - self.sol_colored
        delta_zero    = zero_now    - self.sol_zero
        alpha = 1.0   # weight for new colored nodes
        beta  = 1.0   # weight for new zero nodes (penalty strength)

        
        # to ensure min edge weights have uncolorable nodes
        x = self.x
        g = self.g
        src, dst = g.edges()
        w = g.edata['weight'].squeeze() 
        
        zero_now_mask = (x == 0)
        prev_zero_mask = self.prev_zero_mask
        new_zero_mask = zero_now_mask & (~prev_zero_mask)
        new_zero_nodes = new_zero_mask.nonzero(as_tuple=True)[0]
        min_w_per_new_zero = []
        if new_zero_nodes.numel() > 0:
            for u in new_zero_nodes.tolist():
                mask_incident = (src == u) | (dst == u)
                if not mask_incident.any():
                    continue
                inc_src = src[mask_incident]
                inc_dst = dst[mask_incident]
                inc_w   = w[mask_incident]
                min_w = inc_w.min()
                neighbors = torch.where(inc_src == u, inc_dst, inc_src)
                neighbor_colors = x[neighbors]
                valid_neighbor_mask = ((neighbor_colors >= 1) & (neighbor_colors <= self.k)).squeeze(-1) #original shape:[edges,num_samples], after squeeze :[edges]
                if not valid_neighbor_mask.any():
                    continue
                # print("inc_w.shape:", inc_w.shape)
                # print("valid_neighbor_mask.shape:", valid_neighbor_mask.shape)
                valid_w = inc_w[valid_neighbor_mask]
                min_w = valid_w.min()
                min_w_per_new_zero.append(min_w)
        
        if len(min_w_per_new_zero) > 0:
            min_w_tensor = torch.stack(min_w_per_new_zero)       # [N_new_zero]
            # example: use average of min weights as penalty signal
            avg_w_penalty = min_w_tensor.mean()
        else:
            avg_w_penalty = torch.tensor(0.0, device=self.device)

        self.prev_zero_mask = zero_now_mask

        gamma1 = 0.25
        reward = alpha * delta_colored - beta * delta_zero - gamma1*avg_w_penalty

        self.sol_colored = colored_now
        self.sol_zero    = zero_now

        reward /= self.max_num_nodes

        return reward, self.sol_colored, self.sol_zero, done
    

    def _check_done(self): 
        undecided = (self.x == self.k+1).float()
        self.g.ndata['h'] = undecided
        num_undecided = dgl.sum_nodes(self.g, 'h')
        self.g.ndata.pop('h')
        done = (num_undecided == 0)
            
        return done
    
    def _build_ob(self):
        ob_x = self.x.unsqueeze(2).float()
        ob_t = self.t.unsqueeze(2).float() / self.max_epi_t
        ob = torch.cat([ob_x, ob_t], dim = 2)
        return ob
    
    def register(self, g, num_samples = 1):
        self.g = g
        self.num_samples = num_samples
        self.g.set_n_initializer(dgl.init.zero_initializer)
        self.g.to(self.device)
        self.batch_num_nodes = torch.LongTensor(
            self.g.batch_num_nodes()
            ).to(self.device)
        num_nodes = self.g.number_of_nodes()
        self.x = torch.full(
            (num_nodes, num_samples),
            self.k+1, 
            dtype = torch.long, 
            device = self.device
            )
        self.t = torch.zeros(
            num_nodes, 
            num_samples, 
            dtype = torch.long, 
            device = self.device
            )
        ob = self._build_ob()
        self.prev_zero_mask = torch.zeros_like(self.x, dtype=torch.bool)
        self.sol_colored = torch.zeros(
            self.g.batch_size,     
            num_samples, 
            device=self.device
            )
        self.sol_zero = torch.zeros(
            self.g.batch_size,      
            num_samples, 
            device=self.device
            )
        self.epi_t = torch.zeros(
            self.g.batch_size, 
            num_samples, 
            device = self.device
            )            
            
        return ob