import torch
from scipy.stats import invgamma
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.stats import multivariate_normal
import numpy as np


def multivariate_normal_vector(Zk, mu, S_inv, denominator):
    diff = (Zk - mu).unsqueeze(2)
    diff_t = torch.transpose(diff, 1, 2)
    mult1 = torch.matmul(diff_t, S_inv)
    power = torch.matmul(mult1, diff)[:, 0, 0]
    p_x = torch.exp( -0.5 * power) / denominator
    return p_x

class Model:
    def __init__(self, parameters, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        #  basic parameters
        self.x_dim = 6  # dimension of state vector
        self.z_dim = 2  # dimension of observation vector

        self.Po = 20
        self.H_upd = 100
        #  dynamical model parameters (CV model)
        tau = parameters['tau']
        Q = parameters['Q']
        self.F = torch.tensor([[1, 0, tau, 0, 0, 0],
                               [0, 1, 0, tau, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]]).float().to(device)

        self.F_T = torch.transpose(self.F, 0, 1)

        self.Q = torch.tensor([[tau**4 / 4.0 * Q, 0, tau**3 / 2.0 * Q, 0, 0, 0],
                                [0, tau**4 / 4.0 * Q, 0, tau**3 / 2.0 * Q, 0, 0],
                                [tau**3 / 2.0 * Q, 0, tau**2 * Q, 0, 0, 0],
                                [0, tau**3 / 2.0 * Q, 0, tau**2 * Q, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]
                                ]).float().to(device)

        # survival/death parameters
        self.P_S = .95
        self.Q_S = 1 - self.P_S

        #  birth parameters (LMB birth model, single component only)
        self.T_birth= 4        # no. of LMB birth terms
        # Birth Rate
        self.r_birth = 0.90

        #  observation model parameters (noisy x/y only)
        self.H = torch.tensor([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]
                      ]).float().to(device)

        self.H_T = torch.transpose(self.H, 0, 1)

        R = parameters['R']
        self.R = torch.tensor([[R, 0, 0, 0],
                           [0, R, 0, 0],
                           [0, 0, R, 0],
                           [0, 0, 0, R]]).to(device)  # Measurement Noise

        #  detection parameters
        self.P_D = parameters['P_D'] # 0.98  # probability of detection in measurements
        self.Q_D = 1 - self.P_D

        # clutter parameters
        self.lambda_c = 30                                        # poisson average rate of uniform clutter (per scan)
        self.range_c = None                                      # [ -1000 1000; -1000 1000 ]  uniform clutter region
        self.pdf_c = 0.00000025                                        # 1/prod(self.range_c(:,2)-self.range_c(:,1))  uniform clutter density


class Filter:
    def __init__(self, model):
        # filter parameters
        self.H_upd = 100                  # requested number of updated components/hypotheses
        self.H_max = 500                  # cap on number of posterior components/hypotheses
        self.hyp_threshold = 1e-15           # pruning threshold for components/hypotheses

        self.L_max = 100                   # limit on number of Gaussians in each track - not implemented yet
        self.elim_threshold = 1e-5        # pruning threshold for Gaussians in each track - not implemented yet
        self.merge_threshold = 4          # merging threshold for Gaussians in each track - not implemented yet

        self.z_dim = 2

        self.P_G = 0.9999999                              # gate size in percentage
        self.gamma = invgamma(self.P_G, model.z_dim)      # inv chi^2 dn gamma value
        self.gate_flag = 1                                # gating on or off 1/0

        self.Po = 5


class Track_List_SMC:
    def __init__(self, model=None, N_init=None, device=None, N_particles=None, r_bernoulli=0.9):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if(N_init is None):
            self.Xs = torch.zeros([], device=device)                               # Gaussian Means                            | mus is tensor of shape [Num_Ts x N_Particles x 6]
            self.ws = torch.tensor([], device=device)                                # weights of Gaussians                      | ws is tensor of shape [Num_Ts x N_particles x 1]
            self.Ls = torch.tensor([], device=device)                                # track labels                              | Ls is list of lists. len(Ls) = Num_Ts
            self.Es = [None]                                         # track association history                 | Es is a list of lists. Len(Es) = Num_Ts
            self.Num_Ts = 0                                           # Number of tracks
            self.r = torch.tensor([], device=device)
        else:
            self.Xs = torch.zeros((N_init, N_particles, model.x_dim), device=device)
            self.ws = torch.zeros((N_init, N_particles, 1), device=device)
            self.Ls = [None] * N_init
            self.Es = [None] * N_init
            self.Num_Ts = N_init
            self.r = torch.ones((N_init, 1), device=device) * r_bernoulli

        self.Gated_m = torch.tensor([])                # Gated Measurements

    def get_birth_particles_smc(self, Z, current_label=1, k=0,
                                distance_thres=100,
                                N_particles_per_birth_object=20):
        '''
            Generate Birth Objects from previous measurements that were not labeled
            returns arrays:
                Xs: [N_birth_components, N_particle_per_object, 6]
                ws: [N_birth_components, N_particle_per_object, 1]
        '''


        device = Z.device

        # X_pred = np.array(X_pred)

        absolute_indexes = []

        num_components = self.Num_Ts
        X_pred = torch.zeros((self.Num_Ts, 6), device=device)
        for i in range(num_components):
            X_pred[i, :] = (self.Xs[i, :, :] * self.ws[i, ...]).sum(dim=0)

        if(len(X_pred.shape) > 0 and Z.shape[0] > 0):
            # Do measurement - component association
            cost = torch.cdist(Z[:, 0:2], X_pred[:, 0:2], p=2)
            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
            distances = cost[row_ind, col_ind]
            reduced_indexes = (distances < distance_thres).nonzero()[:, 0]
            absolute_indexes = row_ind[reduced_indexes.cpu().numpy()]

        label = current_label
        H_birth = 0
        new_mu = []
        new_w = []

        L_birth = []

        for i in range(len(Z)):
            if i in absolute_indexes:
                continue

            z = Z[i, :]
            new_z = torch.tensor([z[0].cpu().item(), z[1].cpu().item(), 0, 0, 5, 5]).unsqueeze(0)

            partickes_for_z = generate_N_random_particles(mu=new_z, std=[5.0, 5.0, 2.0, 2.0,  0.0, 0.0], N_particles=N_particles_per_birth_object)
            new_w.append(torch.tensor(torch.ones(N_particles_per_birth_object) / float(N_particles_per_birth_object)).unsqueeze(-1).to(device))

            new_mu.append(partickes_for_z)
            L_birth.append([k, label])
            label += 1
            H_birth += 1

        max_label = label
        if len(new_w):
            w_birth = torch.stack(new_w)
            Xs_birth = torch.stack(new_mu).to(device)
            num_components, num_particles, num_dims = Xs_birth.shape
            r_birth = torch.ones((num_components, 1), device=Xs_birth.device) * 0.9 # hardcoded r_birth
        else:
            w_birth = torch.tensor([], device=device)
            Xs_birth = torch.tensor([], device=device)
            r_birth = torch.tensor([], device=device)

        if len(new_mu) == 0:
            Xs_birth = Xs_birth.unsqueeze(-1)

        tt_birth = Track_List_SMC()
        tt_birth.ws = w_birth
        tt_birth.Xs = Xs_birth
        tt_birth.Ls = L_birth
        tt_birth.r = r_birth
        tt_birth.Es = [None] * H_birth
        tt_birth.Num_Ts = H_birth

        if Z.shape[0] > 0:
            surviving_measurements = Z[absolute_indexes, :]
        else:
            surviving_measurements = torch.tensor([], device=Xs_birth.device)

        return tt_birth, surviving_measurements, max_label


    def kalman_prediction(self, F):
        '''
            Advance State according to F (F transpose) and Q matrices
        '''
        if(self.Num_Ts > 0):
            # weight = np.repeat(weight[:, np.newaxis], x.shape[1], axis=1)
            num_coms, num_parts, model_dims = self.Xs.shape
            temp_Xs = torch.reshape(self.Xs, (num_coms * num_parts, model_dims, 1))
            weight = torch.tensor([4, 4, 2, 2, 0.0, 0.0], device=temp_Xs.device).repeat(num_coms * num_parts, 1)
            #weight = torch.tensor([2, 2, 0.5, 0.5, 0.0, 0.0], device=temp_Xs.device).repeat(num_coms * num_parts, 1)
            weighted_perturbation = torch.randn(temp_Xs.shape[0], temp_Xs.shape[1], device=temp_Xs.device) * weight

            temp_Xs = torch.matmul(F, temp_Xs) + weighted_perturbation.unsqueeze(2)
            self.Xs = torch.reshape(temp_Xs, (num_coms, num_parts, model_dims))


    def kalman_update_3(self, energy_image, regular_image=None):
        '''
            Update State According to model parameters
        '''
        if self.Num_Ts == 0:
            return
        num_components, num_particles, num_dims = self.Xs.shape
        locations = torch.reshape(self.Xs[:, :, 0:2], (num_components * num_particles, 2)).long()

        # Clamp locations to image dimensions
        locations[:, 0] = torch.clamp(locations[:, 0], min=0, max=energy_image.shape[0] - 1)
        locations[:, 1] = torch.clamp(locations[:, 1], min=0, max=energy_image.shape[1] - 1)

        # Get Special Likelihood
        g_x_y = energy_image[locations[:, 0], locations[:, 1]]
        g_x_y = torch.reshape(g_x_y, (num_components, num_particles, 1))

        # update and normalize weights
        scaled_ws = self.ws * g_x_y
        qk = torch.sum(scaled_ws, dim=1) + 0.000000000001

        self.ws = (scaled_ws[:, :, 0] / qk).unsqueeze(2) # ws is [num_components, num_particles, 1] (sums to 1 across num_particles)

        # update and normalize rs
        scaled_rs = self.r * qk
        self.r = scaled_rs / (1 - self.r + scaled_rs)
        return


    def appearance_update(self, pred_tracks, Zk, apps_fts_k):
        '''
            Update State According to high level features
        '''

        device = Zk.device
        Num_Ts_pred = pred_tracks.Num_Ts
        ms = Zk.shape[0]
        sigma_2 = 4.0
        appearance_cost = torch.ones((Num_Ts_pred, ms), device=device) * (1.0 / (np.sqrt(2 * np.pi * sigma_2)) * np.exp(-1.0 / (2.0 * sigma_2)))

        rows, cols, chans = apps_fts_k.shape
        for i in range(Num_Ts_pred):
            pred_i, pred_j, pred_w, pred_h = pred_tracks.mus[i, [0, 1, 4, 5], 0].int()
            pred_w = 3
            pred_h = 3
            # index appearance matrix (5 x 5 x 512)
            if(pred_i + pred_h >= rows or pred_j + pred_w >= cols or pred_i < 0 or pred_j < 0):
                continue
            pred_vector = apps_fts_k[pred_i:pred_i + pred_w, pred_j:pred_j + pred_h, :].flatten()
            pred_vector = pred_vector / torch.norm(pred_vector) # appearance_vector = 1 x 12800
            for j in pred_tracks.Gated_m[i]:
                meas_i, meas_j, meas_w, meas_h = Zk[j, :].int()
                if(meas_i + meas_h >= rows or meas_j + meas_w >= cols):
                    continue
                meas_h = 3
                meas_w = 3
                meas_vector = apps_fts_k[meas_i:meas_i + meas_w, meas_j:meas_j + meas_h, :].flatten()
                meas_vector = meas_vector / torch.norm(meas_vector)
                dot_prod = torch.dot(meas_vector, pred_vector)
                s_i_j = 1.0 - dot_prod
                theta_i_j = 1.0 / (np.sqrt(2 * np.pi * sigma_2)) * torch.exp(-s_i_j / (2.0 * sigma_2))
                appearance_cost[i, j] = theta_i_j
        return appearance_cost

    def kalman_update(self, pred_tracks, model, Zk):
        '''
            Update State According to model parameters
        '''
        device = Zk.device
        # ADD MISDETECTION TRACKS
        for tabidx in range(pred_tracks.Num_Ts):
            self.mus[tabidx] = pred_tracks.mus[tabidx]
            self.Ps[tabidx] = pred_tracks.Ps[tabidx]
            self.ws[tabidx] = pred_tracks.ws[tabidx]
            self.Ls[tabidx] = pred_tracks.Ls[tabidx]

            if(pred_tracks.Es[tabidx] is None):
                self.Es[tabidx] = [-1]
            else:
                self.Es[tabidx] = pred_tracks.Es[tabidx] + [-1]  # -1 means undetected track

        # ADD UPDATED WEIGHTS
        H = model.H
        H_T = model.H_T

        if(pred_tracks.mus.shape[0] == 0):
            return torch.zeros((0, Zk.shape[0]))
        z_pred = torch.matmul(model.H, pred_tracks.mus)                                              # Num_Ts_pred x Z_dim x 1
        S_pred  = torch.matmul(torch.matmul(model.H, pred_tracks.Ps), model.H_T) + model.R          # Num_Ts_pred x Z_dim x Z_dim

        S_inv = torch.inverse(S_pred)                                                                # Num_Ts_pred x Z_dim x Z_dim
        P_H = torch.matmul(pred_tracks.Ps, model.H_T)                                                # Num_Ts_pred x X_dim x Z_dim
        K_S_inv = torch.matmul(P_H, S_inv)                                                           # Num_Ts_pred x X_dim x Z_dim

        K_gain = torch.matmul(K_S_inv, model.H)                                                     # Num_Ts_pred x X_dim x X_dim
        P_updt = torch.matmul((torch.eye(model.H.shape[1], device=device) - K_gain), pred_tracks.Ps)               # Num_Ts_pred x X_dim x X_dim


        Num_Ts_pred = pred_tracks.Num_Ts
        ms = Zk.shape[0]
        Zk = torch.tensor(Zk)
        allcostm = torch.zeros((Num_Ts_pred, ms), device=device)

        for i in range(Num_Ts_pred):
            for j in pred_tracks.Gated_m[i]:
                w_temp = pred_tracks.ws[i, :] * torch.tensor(multivariate_normal.pdf(Zk[j, :].cpu(), mean=z_pred[i, :, 0].cpu(), cov=S_pred[i, :, :].cpu()), device=device).unsqueeze(-1)
                mu_temp = pred_tracks.mus[i] + torch.matmul(K_S_inv[i, :, :], (Zk[j, :].float() - z_pred[i, :, 0])).unsqueeze(-1)                        # X_dim x 1
                P_temp = P_updt[i, :, :]

                # Populated updated tracks
                stoidx = Num_Ts_pred * (j + 1) + i
                self.mus[stoidx] = mu_temp
                self.Ps[stoidx] = P_temp
                self.ws[stoidx] = w_temp / (w_temp + 0.00000001)
                self.Ls[stoidx] = pred_tracks.Ls[i]
                if(pred_tracks.Es[i] is None):
                    self.Es[stoidx] = [j.item()]
                else:
                    self.Es[stoidx] = pred_tracks.Es[i] + [j.item()]

                allcostm[i, j] = w_temp
                # self.ws[i, :] = w_temp / (w_temp + 0.00000001)

        allcostm = allcostm
        return allcostm

    def get_gated_measurements(self, Zk, distance_thres=50, perform_gating=True):
        '''
            Update Gated Measurements (TO DO)
        '''
        if len(self.Xs.shape) > 0 and Zk.shape[0] > 0 and perform_gating:
            mus = self.Xs.mean(dim=1)
            # Do measurement - component association
            mus = mus[:, 0:2]
            cost = torch.cdist(mus, Zk[:, 0:2], p=2)
            gates = cost < distance_thres
            Gated_m = []
            for com in range(gates.shape[0]):
                gate_indexes = gates[com, :].nonzero()
                if gate_indexes.shape[0] > 0:
                    Gated_m.append(gate_indexes[:, 0])
                else:
                    Gated_m.append(torch.tensor([], device=self.Xs.device))
        else:
            Gated_m = [torch.arange(Zk.shape[0], device=self.Xs.device)] * self.Num_Ts
        self.Gated_m = Gated_m

    def index_w_tensor(self, indexes, wb=1.0):
        new_track_list = Track_List_SMC()
        new_track_list.Xs = self.Xs[indexes, :, :]
        new_track_list.ws = wb * self.ws[indexes]
        new_track_list.Ls = [self.Ls[i] for i in indexes]
        new_track_list.Es = [self.Es[i] for i in indexes]
        new_track_list.Num_Ts = len(new_track_list.Ls)
        return new_track_list

class glmb_smc_instance:
    def __init__(self, track_list=None, device=None):

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # initial prior
        self.tt = Track_List_SMC()                      # track table for GLMB (cell array of structs for individual tracks)
        self.w = torch.tensor([], device=device)                  # vector of GLMB component/hypothesis weights
        self.I = None

        # cell of GLMB component/hypothesis labels (labels are indices/entries in track table)
        self.n = []                                  # vector of GLMB component/hypothesis cardinalities
        self.cdn = []                               # cardinality distribution of GLMB (vector of cardinality distribution probabilities)

        if track_list is not None:
            self.tt = track_list
        else:
            self.w = torch.tensor([1], device=device)
            self.n = [0]
            self.cdn = [1]


    def prune(self, threshold=0.01):
        # prune components with weights lower than specified threshold
        idxkeep = (self.w > threshold).nonzero()[:, 0]

        # Update I SETS
        if self.I is not None:
            self.I = [self.I[i] for i in idxkeep]

        self.w = self.w[idxkeep]
        norm_w = self.w.sum()
        self.w = self.w / norm_w
        self.n = self.n[idxkeep]

        # Recalculate Cardinality Distribution
        temp_cdn = []
        for card in range(self.n.max() + 1):
            card_bin = (self.w[self.n == card]).sum()
            temp_cdn.append(card_bin)

        self.cdn = torch.tensor(temp_cdn)


    def extract_estimates(self):
        '''
            extract estimates via best cardinality, then
            best component/hypothesis given best cardinality, then
            best means of tracks given best component/hypothesis and cardinality
        '''
        Xk = {}
        N = self.tt.Num_Ts

        if N > 0:
            X_inf = torch.zeros((self.tt.Xs.shape[2], N))
            L = torch.zeros((2, N))

            for n in range(N):
                if self.tt.ws[n, ...].sum() == 0:
                    continue
                X_inf[:, n] = (self.tt.Xs[n, :, :] * self.tt.ws[n, ...]).sum(dim=0)
                label = self.tt.Ls[n][1]

                Xk[label] = X_inf[:, n]
        return Xk


    def inference(self, all=False):
        '''

        :param all:
        :return: returns dictionary Dk['Hs'], Dk['mus'], Dk['Ps']...etc
        '''
        if all:
            return self.extract_all_estimates()
        else:
            return self.extract_estimates()


def logsumexp(w):
    val = w.max()
    return torch.log((torch.exp(w - val)).sum()) + val

def mbestwrap_updt_gibbsamp(P0, m, p=None):
    '''
        P0 is of shape [ (see table in paper)]
    '''
    device = P0.device
    n1, n2 = P0.shape
    assignments = torch.zeros(m, n1, device=device)
    costs = torch.zeros(1, m, device=device)
    
    currsoln = torch.arange(n1, 2 * n1, device=device)    # use all missed detections as initial solution
    assignments[0, :] = currsoln

    # in the case of an empty hypothesis
    if n1 == 0:
        return assignments[0, :].unsqueeze(0), costs[:, 0].unsqueeze(1)
    temp_cost = P0[:, currsoln]
    idx_to_gather = torch.arange(0, n1, device=device).unsqueeze(-1)
    temp_cost = torch.gather(temp_cost, dim=1, index=idx_to_gather)
    costs[0, 0] = temp_cost.sum()
    for sol in range(1, m):
        for var in range(n1):
            tempsamp = torch.exp(-P0[var, :])                           # grab row of costs for current association variable
            lock_idxs = torch.cat((currsoln[0:var], currsoln[var + 1:]))
            tempsamp[lock_idxs] = 0 # lock out current and previous iteration step assignments except for the one in question
            idx_old = (tempsamp > 0).nonzero(as_tuple=True)
            tempsamp = tempsamp[idx_old]
            cdf = tempsamp / tempsamp.sum()
            cdf_bins = torch.cat((torch.tensor([0], device=device), torch.cumsum(cdf, dim=0)), 0)
            sample = torch.rand(1, 1, device=device)
            bin_idx = (cdf_bins > sample).nonzero()[0][1] - 1           # Get first idx that is larger than sample (-1 becase we are also counting 0)
            currsoln[var] = idx_old[0][bin_idx]
        assignments[sol, :] = currsoln
        temp_cost = P0[:, currsoln]
        temp_cost = torch.gather(temp_cost, dim=1, index=idx_to_gather)
        costs[0, sol] = temp_cost.sum()

    C, inverse_I = torch.unique(assignments, sorted=True, return_inverse=True, dim=0)
    perm = torch.arange(inverse_I.size(0), device=device)
    inverse_I, perm = inverse_I.flip([0]), perm.flip([0])
    I = inverse_I.new_empty(C.size(0)).scatter_(0, inverse_I, perm)

    assignments = C
    costs = costs[:, I]
    return assignments, costs



'''





def get_birth_particles_smc_image(signal_image, tt_pred, current_label=1, k=0, distance_thres=100, N_particles_per_birth_object=20, device=None):
    if device is None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    absolute_indexes = []

    num_components = tt_pred.Num_Ts
    X_pred = torch.zeros((tt_pred.Num_Ts, 6), device=device)
    for i in range(num_components):
        X_pred[i, :] = (tt_pred.Xs[i, :, :] * tt_pred.ws[i, ...]).sum(dim=0)

    if len(X_pred.shape) > 0 and Z.shape[0] > 0:
        # Do measurement - component association
        cost = torch.cdist(Z[:, 0:2], X_pred[:, 0:2], p=2)
        row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
        distances = cost[row_ind, col_ind]
        reduced_indexes = (distances < distance_thres).nonzero()[:, 0]
        absolute_indexes = row_ind[reduced_indexes.cpu().numpy()]

    label = current_label
    H_birth = 0
    new_mu = []
    new_w = []

    L_birth = []

    for i in range(len(Z)):
        if i in absolute_indexes:
            continue

        z = Z[i, :]
        new_z = torch.tensor([z[0].cpu().item(), z[1].cpu().item(), 0, 0, 5, 5]).unsqueeze(0)
        partickes_for_z = generate_N_random_particles(mu=new_z, std=[10.0, 10.0, 1.0, 1.0,0.0, 0.0], N_particles=N_particles_per_birth_object)
        new_w.append(torch.tensor(torch.ones(N_particles_per_birth_object) / float(N_particles_per_birth_object)).unsqueeze(-1).to(device))

        new_mu.append(partickes_for_z)
        L_birth.append([k, label])
        label += 1
        H_birth += 1

    max_label = label
    if len(new_w):
        w_birth = torch.stack(new_w)
        Xs_birth = torch.stack(new_mu).to(device)
        num_components, num_particles, num_dims = Xs_birth.shape
        r_birth = torch.ones((num_components, 1), device=Xs_birth.device) * 0.9 # hardcoded r_birth
    else:
        w_birth = torch.tensor([], device=device)
        Xs_birth = torch.tensor([], device=device)
        r_birth = torch.tensor([], device=device)

    if len(new_mu) == 0:
        Xs_birth = Xs_birth.unsqueeze(-1)

    tt_birth = Track_List_SMC()
    tt_birth.ws = w_birth
    tt_birth.Xs = Xs_birth
    tt_birth.Ls = L_birth
    tt_birth.r = r_birth
    tt_birth.Es = [None] * H_birth
    tt_birth.Num_Ts = H_birth

    if Z.shape[0] > 0:
        surviving_measurements = Z[absolute_indexes, :]
    else:
        surviving_measurements = torch.tensor([], device=Xs_birth.device)

    return tt_birth, surviving_measurements, max_label
'''



def generate_N_random_particles(mu, std=[2.0, 2.0, 1.0, 1.0,0.0, 0.0], N_particles=20):
    '''
        Generate N random particles at this location
        mu: [1, 6]
        std: list of length(6)

    '''
    torch.manual_seed(0)
    means = torch.zeros((N_particles, 6))
    std = torch.tensor(std)
    random_perturbations = torch.normal(means, std)
    return mu + random_perturbations

def concatenate_tracks_smc(tt_left, tt_right):
    '''
        Append tt_left to tt_right
    '''

    new_tracks = Track_List_SMC()
    if len(tt_right.Xs.shape) == 0 or tt_right.Xs.shape[0] == 0:
        new_tracks = tt_left
        return new_tracks
    if len(tt_left.Xs.shape) == 0 or tt_left.Xs.shape[0] == 0:
        new_tracks = tt_right
        return new_tracks

    new_tracks.Xs = torch.cat((tt_left.Xs, tt_right.Xs), 0)                                 # tensor concat
    new_tracks.ws = torch.cat((tt_left.ws, tt_right.ws), 0)                                    # tensor concat
    new_tracks.Ls = tt_left.Ls + tt_right.Ls                                                   # list concatenation
    new_tracks.Es = tt_left.Es + tt_right.Es                                                   # list concatenation
    new_tracks.Num_Ts = tt_left.Num_Ts + tt_right.Num_Ts                                       # list concatenation
    return new_tracks


def clean_update(glmb_smc_clean):

    '''
    # flag used tracks
    usedindicator = torch.zeros(glmb_temp.tt.Num_Ts)
    for hidx in range(len(glmb_temp.w)):
        usedindicator[glmb_temp.I[hidx].long()] = usedindicator[glmb_temp.I[hidx].long()] + 1
    track_indices = usedindicator > 0
    trackcount = track_indices.sum()

    # remove unused tracks and reindex existing hypotheses/components
    newindices = torch.zeros(glmb_temp.tt.Num_Ts)
    newindices[track_indices] = torch.arange(trackcount, dtype=newindices.dtype)

    cleaned_track_list = glmb_temp.tt.index_w_tensor(track_indices.nonzero()[:, 0])
    glmb_smc_clean = glmb_smc_instance(track_list=cleaned_track_list)
    glmb_smc_clean.w = glmb_temp.w
    glmb_smc_clean.n = glmb_temp.n
    glmb_smc_clean.cdn = glmb_temp.cdn

    for hidx in range(len(glmb_temp.w)):
        if(hidx == 0):
            glmb_smc_clean.I = [newindices[glmb_temp.I[hidx].long()]]
        else:
            glmb_smc_clean.I.append(newindices[glmb_temp.I[hidx].long()])
    '''
    # resampling step
    indexes_to_keep = []
    for tabid in range(glmb_smc_clean.tt.Num_Ts):
        cmpt_weights = glmb_smc_clean.tt.ws[tabid, ...]
        neffsample = 1.0 / ((cmpt_weights ** 2).sum() + 0.0000001)
        if neffsample < 10000:
            X_temp = glmb_smc_clean.tt.Xs[tabid, ...]
            w_temp = glmb_smc_clean.tt.ws[tabid, ...]
            idxs = torch.utils.data.WeightedRandomSampler(w_temp[:, 0], w_temp.shape[0])
            glmb_smc_clean.tt.Xs[tabid, ...] = X_temp[list(idxs), :]
            new_weights = torch.ones((len(idxs), 1), device=w_temp.device) / w_temp.shape[0] # w_temp[list(idxs), ...]
            glmb_smc_clean.tt.ws[tabid, ...] = new_weights
            indexes_to_keep.append(tabid)
        else:
            pass

    glmb_smc_clean.tt = glmb_smc_clean.tt.index_w_tensor(indexes_to_keep)

    return glmb_smc_clean

def clean_predict(glmb_raw):
    # hash label sets, find unique ones, merge all duplicates
    return glmb_raw


def prune(glmb_updated):
    # prune components with weights lower than specified threshold
    idxkeep = (glmb_updated.w > 0.01).nonzero()[:, 0]
    glmb_pruned = glmb_smc_instance(track_list=glmb_updated.tt)
    glmb_pruned.w = glmb_updated.w[idxkeep]
    if(glmb_updated.I is not None):
        glmb_pruned.I = [glmb_updated.I[i] for i in idxkeep]
    else:
        glmb_pruned.I = None
    glmb_pruned.n = glmb_updated.n[idxkeep]

    glmb_pruned.w = glmb_pruned.w / glmb_pruned.w.sum()

    for card in range(glmb_pruned.n.max() + 1):
        card_bin = (glmb_pruned.w[glmb_pruned.n == card]).sum()
        glmb_pruned.cdn.append(card_bin)

    glmb_pruned.cdn = torch.tensor(glmb_pruned.cdn)

    return glmb_pruned



def extract_all_possible_tracks(glmb_update):
    '''
        extract estimates via best cardinality, then
        best component/hypothesis given best cardinality, then
        best means of tracks given best component/hypothesis and cardinality
    '''
    Dk = {}
    Dk['mus'] = []
    Dk['ws'] = []
    Dk['Ps'] = []
    Dk['Ls'] = []
    Dk['Hs'] = 0

    N = torch.argmax(glmb_update.cdn)
    if(glmb_update.tt.Num_Ts == 0):
        return Dk

    for track_index in range(glmb_update.tt.Num_Ts):
        Dk['mus'].append(glmb_update.tt.mus[track_index, :, 0].numpy())
        Dk['Ls'].append(torch.tensor(glmb_update.tt.Ls[track_index])[1].item())
        Dk['ws'].append(1)
        Dk['Ps'].append(torch.tensor(glmb_update.tt.Ps[track_index]))

    Dk['Hs'] = N
    return Dk


def extract_estimates_LMB_only(glmb_update):
    '''
        extract estimates via best cardinality, then
        best component/hypothesis given best cardinality, then
        best means of tracks given best component/hypothesis and cardinality
    '''
    Xk = {}
    N = glmb_update.tt.Num_Ts

    if N > 0:
        X_inf = torch.zeros((glmb_update.tt.Xs.shape[2], N))
        L = torch.zeros((2, N))

        for n in range(N):
            if glmb_update.tt.ws[n, ...].sum() == 0:
                continue
            X_inf[:, n] = (glmb_update.tt.Xs[n, :, :] * glmb_update.tt.ws[n, ...]).sum(dim=0)
            label = glmb_update.tt.Ls[n][1]

            Xk[label] = X_inf[:, n]
    return Xk



def extract_estimates(glmb_update):
    '''
        extract estimates via best cardinality, then
        best component/hypothesis given best cardinality, then
        best means of tracks given best component/hypothesis and cardinality
    '''
    Dk = {}
    Dk['mus'] = []
    Dk['ws'] = []
    Dk['Ls'] = []
    Dk['Ps'] = []
    Dk['Hs'] = 0

    N = torch.argmax(glmb_update.cdn)
    if glmb_update.tt.Num_Ts == 0:
        return Dk

    print("Starting the Estimate Extraction")
    X_inf = torch.zeros((glmb_update.tt.Xs.shape[2], N))
    L = torch.zeros((2, N))

    idxcmp = torch.argmax(glmb_update.w * (glmb_update.n == N))

    for n in range(N):
        track_index = glmb_update.I[idxcmp][n].long()

        X_inf[:, n] = (glmb_update.tt.Xs[track_index, :, :] * glmb_update.tt.ws[track_index, ...]).sum(dim=0)
        L[:, n] = torch.tensor(glmb_update.tt.Ls[track_index])

        Dk['mus'].append(X_inf[:, n].cpu().numpy())
        Dk['Ls'].append(torch.tensor(glmb_update.tt.Ls[track_index])[1].item())
        Dk['ws'].append(1)
        Dk['Ps'].append(torch.tensor([[1, 0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 1]]))

    Dk['Hs'] = N
    return Dk


def extract_all_estimates(glmb_update):
    '''
        extract estimates via best cardinality, then
        best component/hypothesis given best cardinality, then
        best means of tracks given best component/hypothesis and cardinality
    '''
    Dk = {}
    Dk['mus'] = []
    Dk['ws'] = []
    Dk['Ps'] = []
    Dk['Ls'] = []
    Dk['Hs'] = 0

    N = torch.argmax(glmb_update.cdn)
    if(glmb_update.tt.Num_Ts == 0):
        return Dk

    for track_index in range(glmb_update.tt.Num_Ts):
        track_index = track_index

        Dk['mus'].append(glmb_update.tt.mus[track_index, :, 0].cpu().numpy())
        Dk['Ls'].append(torch.tensor(glmb_update.tt.Ls[track_index])[1].item())
        Dk['ws'].append(1)
        Dk['Ps'].append(torch.tensor(glmb_update.tt.Ps[track_index]))

    Dk['Hs'] = glmb_update.tt.Num_Ts
    return Dk


def lmb2glmb(tt_lmb, H_req):
    # convert LMB to GLMB
    r_vector = tt_lmb.r
    cost_vector = r_vector / (1 - r_vector)
    neglogcostv= -torch.log(cost_vector)
    # print("ok")


def kshortest_paths(rs, k):
    print("ok")



class priorityQ_torch(object):
    """Priority Q implelmentation in PyTorch

    Args:
        object ([torch.Tensor]): [The Queue to work on]
    """

    def __init__(self, val):
        self.q = torch.tensor([[val, 0]])
        # self.top = self.q[0]
        # self.isEmpty = self.q.shape[0] == 0

    def push(self, x):
        """Pushes x to q based on weightvalue in x. Maintains ascending order

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]
            x ([torch.Tensor]): [[index, weight] tensor to be inserted]

        Returns:
            [torch.Tensor]: [The queue tensor after correct insertion]
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        if self.isEmpty():
            self.q = x
            self.q = torch.unsqueeze(self.q, dim=0)
            return
        idx = torch.searchsorted(self.q.T[1], x[1])
        print(idx)
        self.q = torch.vstack([self.q[0:idx], x, self.q[idx:]]).contiguous()

    def top(self):
        """Returns the top element from the queue

        Returns:
            [torch.Tensor]: [top element]
        """
        return self.q[0]

    def pop(self):
        """pops(without return) the highest priority element with the minimum weight

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]

        Returns:
            [torch.Tensor]: [highest priority element]
        """
        if self.isEmpty():
            print("Can Not Pop")
        self.q = self.q[1:]

    def isEmpty(self):
        """Checks is the priority queue is empty

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]

        Returns:
            [Bool] : [Returns True is empty]
        """
        return self.q.shape[0] == 0


def dijkstra(adj):
    n = adj.shape[0]
    distance_matrix = torch.zeros([n, n])
    for i in range(n):
        u = torch.zeros(n, dtype=torch.bool)
        d = np.inf * torch.ones(n)
        d[i] = 0
        q = priorityQ_torch(i)
        while not q.isEmpty():
            v, d_v = q.top()  # point and distance
            v = v.int()
            q.pop()
            if d_v != d[v]:
                continue
            for j, py in enumerate(adj[v]):
                if py == 0 and j != v:
                    continue
                else:
                    to = j
                    weight = py
                    if d[v] + py < d[to]:
                        d[to] = d[v] + py
                        q.push(torch.Tensor([to, d[to]]))
        distance_matrix[i] = d
    return distance_matrix
