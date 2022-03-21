import numpy as np
from scipy.integrate import nquad
from scipy.special import erfc, erfcx
from scipy.stats import norm
from scipy.integrate import quad_vec, quad
from sklearn.datasets import make_spd_matrix
from nb_integrals import I0_00, I1_00, I2_00, I0_10, I1_10, I2_10
import time
import sys
np.random.seed(0)

class AMP():

    def __init__(self, n_samples, n_labels, alpha, var_noise= 0.0, channel= 'argmax', prior= 'gauss', regul= 1., damping= 0., tol= 1e-10, infinity= 10.):
        """Initialization"""
        # Problem dimensions and inputs
        self.n = n_samples
        self.k = n_labels-1
        self.d = int(round(n_samples*alpha))
        self.var_noise = var_noise
        self.channel = channel
        self.alpha = alpha
        self.prior = prior
        self.diff_W_ = []
        self.ovlap_matrix_ = []
        self.prior_reg = regul
        self.tol = tol
        self.infinity = infinity
        self.damping = damping
        self.f_teacher = None
        self.noise = None
        # Setting the parameters for the unreduced problem
        self.k0 = self.k + 1
        self.K0 = 2**(self.k+1)


        """Initialize AMP outputs"""

        if prior == 'gauss':
            self.W_hat =  np.random.standard_normal((self.n, self.k)) / self.n

        if prior == 'rademacher':
            # Possible configurations
            config = []
            for l in range(self.K0):
                config.append(2.*np.array( [int(x) for x in list('{0:0b}'.format(l).zfill(self.k0))])-1.)
            config = np.array(config)
            # Balancing
            for l in range(self.k):
                config[:, l] = config[:, l] +  config[:, self.k0-1]
            configs = np.delete(config, self.k0-1, axis=1)
            rng = np.random.default_rng()
            self.W_hat = rng.choice(configs, self.n, replace=True) / self.n

        self.C_hat = np.array([make_spd_matrix(self.k) for j in range(self.n)]) / self.k

        """Placeholder"""
        self.omega = np.zeros((self.d, self.k))
        self.V = np.ones((self.d, self.k, self.k))

        """Covariance for the gaussian prior"""
        self.cov_inv = np.linalg.inv(np.ones(self.k) + np.eye(self.k))


    def data(self, teacher= 'argmax', beta= 1.0):
        """Defines teacher and generates data"""
        """teacher = argmax or softmax"""
        # Gaussian random teacher weights
        if self.prior == 'gauss':
            cov = np.ones(self.k) + np.eye(self.k)
            self.W_star = np.random.multivariate_normal(mean=np.zeros(self.k), cov= cov, size= self.n)
        if self.prior == 'rademacher':
            config = []
            for l in range(self.K0):
                config.append(2.*np.array( [int(x) for x in list('{0:0b}'.format(l).zfill(self.k0))])-1.)
            config = np.array(config)
            # Balancing
            for l in range(self.k):
                config[:, l] = config[:, l] +  config[:, self.k0-1]
            self.config = np.delete(config, self.k0-1, axis=1)
            self.config2 = np.einsum('ij,ik-> ijk', self.config, self.config)
            id_ = np.random.choice(self.config.shape[0], size= self.n, replace=True)
            self.W_star = self.config[id_]

        # Gaussian random data
        self.X = np.random.randn(self.d, self.n) / np.sqrt(self.n)
        self.Xsq = np.square(self.X)
        # Noise
        self.noise = np.sqrt(self.var_noise) * np.random.randn(self.d, self.k)
        # Teacher labels
        if teacher == 'softmax':
            self.y = self.softmax(np.matmul(self.X, self.W_star) + self.noise, beta)
            self.f_teacher = self.softmax
        if teacher == 'argmax':
            self.y = self.argmax(np.matmul(self.X, self.W_star) + self.noise)
            self.f_teacher = self.argmax
        if teacher == 'linear':
            self.y = self.linear(np.matmul(self.X, self.W_star) + self.noise)
            self.f_teacher = self.linear

        print('  ')
        print('| | | | | AMP | | | | |')
        print(' ')
        print('--- Teacher weights ---')
        print('W_star= ', self.W_star)
        print(' ')

        return self.X, self.y, self.W_star


    def linear(self, x):
        return x


    def softmax(self, x, beta=1.):
        """Compute softmax values"""
        e_x = np.exp(beta*(x - np.max(x)))
        return e_x / e_x.sum(axis=1)[:,None]


    def argmax(self, x):
        """Compute argmax for each row"""
        y = np.zeros_like(x)
        y[np.arange(len(x)), x.argmax(1)] = (x.max(axis = 1)>0).astype(int)
        return y


    def eg(self, new_samples):
        """Compute generalization error"""
        new_samples = int(round(new_samples))
        X_new = np.random.randn(new_samples, self.n) / np.sqrt(self.n)
        noise = np.sqrt(self.var_noise) * np.random.randn(new_samples, self.k)
        ynew_t = self.f_teacher(np.matmul(X_new, self.W_star) + noise)
        ynew_s = self.f_teacher(np.matmul(X_new, self.W_hat) + noise)

        ynew_t = np.insert(ynew_t, ynew_t.shape[1], np.abs(np.sum(ynew_t, axis=1)-1.), axis=1)
        ynew_s = np.insert(ynew_s, ynew_s.shape[1], np.abs(np.sum(ynew_s, axis=1)-1.), axis=1)
        error = 0.5*np.mean(np.sum((ynew_s - ynew_t)**2, axis=1))
        return error


    def _damping(self, X, X_old):
        y = (1.- self.damping)*X + self.damping*X_old
        return y


    def fit(self, max_iter=1000, conv=1e-8, W0_noise= 0.):
        """Run AMP"""
        ## Define channel and prior
        if self.prior == 'gauss':
            f_prior = self.f_prior_gauss
        elif self.prior == 'rademacher':
            if self.k == 1:
                f_prior = self.f_prior_radem_k1
            else:
                f_prior = self.f_prior_radem_k2
        if self.channel == 'quad':
            f_channel = self.f_channel_quad
        elif self.channel == 'argmax':
            if self.k == 2:
                f_channel = self.f_ch_argmax_k2
            else:
                print('argmax')
                f_channel = self.f_ch_argmax_k1
        elif self.channel == 'probit':
            if self.k == 1:
                print('probit')
                f_channel = self.f_ch_probit

        ## Convergence flag
        conv_tol = False

        ## Initialization close to the solution to check stability
        print('--- Initialization ---')
        if W0_noise > 0:
            print('W_hat0 ~ W_star + gaussian with variance %.12f' % W0_noise)
            self.W_hat = self.W_star + np.sqrt(W0_noise)*np.random.standard_normal(self.W_star.shape)
            ov0 = np.dot(self.W_star.T, self.W_hat) / self.n
        else:
            print('W_hat0 ~ ' + self.prior)
            ov0 = np.dot(self.W_star.T, self.W_hat) / self.n

        print('initial overlap = ', ov0)
        print('   ')

        self.ovlap_matrix_.append(ov0)

        ## Initialize channel and prior matrice
        f_out = np.zeros((self.d, self.k))  / self.n
        df_out = np.ones((self.d, self.k, self.k)) / self.k

        print('--- Iterate AMP - alpha = %.8f ---' % self.alpha)
        ## Iterate AMP
        t0_total = time.time()
        for t in range(max_iter):
            t0 = time.time()
            # Channel: update the mean omega and variance V
            self.V = np.einsum('ijk,li->ljk', self.C_hat, self.Xsq)
            omega1 = np.einsum('ij, jk->ik', self.X, self.W_hat)
            omega2 = np.einsum('ijk,ik -> ij', self.V, f_out)
            self.omega = omega1 - omega2

            self.f_out_old = np.copy(f_out)
            self.df_out_old = np.copy(df_out)
            # Update: f_out and df_out
            f_out, df_out = f_channel(self.y, self.omega, self.V)

            if bool(self.damping):
                f_out = self._damping(f_out, self.f_out_old)
                df_out_old = self._damping(df_out, self.df_out_old)

            # Prior: update the mean gamma and the variance Lambda
            Lambda = -np.einsum('ij, ikl -> jkl', self.Xsq, df_out)
            gamma1 = np.einsum('ij, ik -> jk', self.X, f_out)
            gamma2 = np.einsum('ijk, ik -> ij', Lambda, self.W_hat)
            gamma = gamma1 + gamma2

            W_hat_old = np.copy(self.W_hat)
            C_hat_old = np.copy(self.C_hat)
            # Update the estimate marginals
            self.W_hat, self.C_hat = f_prior(gamma, Lambda)

            # Damping
            if bool(self.damping):
                self.W_hat = self._damping(self.W_hat, W_hat_old)
                self.C_hat = self._damping(self.C_hat, C_hat_old)

            # Metrics
            diff_W = np.mean(np.abs(W_hat_old - self.W_hat))
            mses_W = np.mean((self.W_hat - self.W_star)**2)
            self.diff_W_ .append(diff_W)
            ov = np.dot(self.W_star.T, self.W_hat) / self.n
            self.ovlap_matrix_.append(ov)
            # Iterarion status
            t1 = time.time()
            if self.k == 1:
                print('alpha= %.8f | it= %d | diff_W= %.8f | time: %.3fs | ov= %.8f' % (self.alpha, t, diff_W, t1-t0, ov))
            else:
                print('alpha= %.8f | it= %d | diff_W= %.8f | mses = %.8f | time: %.3fs ' % (self.alpha, t, diff_W, mses_W, t1-t0))
            if t % 10 == 0:
                print('overlap matrix = ', ov)
            # Check for convergence
            if diff_W < conv:
                conv_tol = True
                break

        t1_total = time.time()
        print('Terminating AMP alpha= %.8f' % self.alpha)

        if conv_tol:
            print('mean(abs(W-W_old)) < %.5f' % conv)
        else:
            print('maximum number of iterations %d achieved' % max_iter)

        mses = np.mean((self.W_hat - self.W_star)**2)
        self.mses = mses
        print('it= %d | mseW= %.8f | diffW= %.8f | total time: %.3fs ' % (t, mses, diff_W, t1_total-t0_total))
        print('final overlap matrix = ', ov)
        print('    ')


    def get_diff_W(self):
        """Returns list of diffW after fit"""
        return self.diff_W_


    def f_prior_gauss(self, gamma, Lambda):
        """f0 and f0' for the Gaussian prior"""
        L  = np.linalg.inv(Lambda + self.prior_reg*self.cov_inv)
        f0 =  np.einsum('ijk,ik -> ij', L , gamma)
        return f0, L


    def f_prior_radem_k1(self, gamma, Lambda):
        """f0 and f0' for the Rademacher prior -- k=1"""
        identity = np.zeros(Lambda.shape)
        np.einsum('jii->ij', identity)[:] = 1
        f0 = np.tanh(gamma)
        df0 = identity - np.einsum('ij,ik -> ijk', f0 , f0)
        return f0, df0


    def f_prior_radem_k2(self, gamma, Lambda):
        """f0 and df0' for the Rademacher prior -- k>2"""
        f0 = np.copy(self.W_hat)
        df0 = np.copy(self.C_hat)
        Z = np.zeros(gamma.shape[0])
        # Replicate for each configuration
        w_ = np.tile(self.config, reps=[gamma.shape[0],1])
        w2_ = np.tile(self.config2, reps=[gamma.shape[0],1,1])
        gamma_ = np.tile(gamma, self.K0).reshape(gamma.shape[0]*self.K0, self.k)
        Lambda_ = np.tile(Lambda, reps=[self.K0,1]).reshape(Lambda.shape[0]*self.K0, self.k, self.k)

        gammaT_w_ = np.einsum('ij,ij->i', gamma_, w_)
        Lambda_w_ = np.einsum('ijk,ij->ik', Lambda_, w_)
        wT_Lambda_w_ =  np.einsum('ij,ij->i', w_, Lambda_w_)
        expo = -0.5*wT_Lambda_w_ + gammaT_w_

        # Convenient shift (makes 0 < Z <=1 )
        expo = expo - np.max(expo)*np.ones(expo.shape)

        expo_ = np.exp(expo)
        wexpo_ = np.einsum('i,ik-> ik', expo_, w_)
        w2expo_ = np.einsum('i,ijk-> ijk', expo_, w2_)

        Z = np.add.reduceat(expo_, np.arange(0, expo_.shape[0], self.K0))
        wexpo = np.add.reduceat(wexpo_, np.arange(0, wexpo_.shape[0], self.K0))
        w2expo = np.add.reduceat(w2expo_, np.arange(0, w2expo_.shape[0], self.K0))
        # After the transition some components have Z -> 0, so we update just Z >0
        zero = 1e-100
        Z_red=  Z[Z>zero]
        wexpo_red=  wexpo[Z>zero]
        w2expo_red =  w2expo[Z>zero]

        f0[Z>zero] = np.einsum('ij,i->ij', wexpo_red, 1./Z_red )
        df0[Z>zero] = np.einsum('ijk,i-> ijk', w2expo_red, 1./Z_red) -  np.einsum('ij,ik -> ijk', f0[Z>zero] , f0[Z>zero])

        return f0, df0


    def f_channel_quad(self, y, omega, V):
        """f_out and df_out for the quadratic loss"""
        identity = np.zeros(V.shape)
        np.einsum('jii->ij', identity)[:] = 1

        V_inv = np.linalg.inv(V)
        V_inv_t = np.einsum('ijk->ikj', V_inv)

        Vs = (V_inv + V_inv_t)/2
        Vs_ = np.linalg.inv(Vs + identity)
        Vs_2 =  np.einsum('ijk,ikl->ijl', Vs_, Vs)

        df =  -np.einsum('ijk,ikl->ijl', V_inv, identity - Vs_2)

        f1 =  np.einsum('ijk,ik -> ij', df, omega)
        V3 = np.einsum('ijk,ikl->ijl', V_inv, Vs_)
        f2 =  np.einsum('ijk,ik -> ij', V3, y)
        f = f1 + f2
        return f, df



    def parameters(self):
        """Estimator parameters"""
        return self.W_hat, self.C_hat


    def _parameter(self):
        """Ground truth"""
        return self.W_star


    def get_mses(self):
        """Returns list of MSES after fit"""
        return self.mses


    def overlap_matrix(self):
        """Returns list of overlap matrices after fit"""
        return self.ovlap_matrix_


    def f_ch_argmax_k2(self, y, omega, V):
        """f_out and df_out for the argmax loss (k = 2)"""

        f = np.zeros((self.d, self.k))
        df = np.zeros((self.d, self.k, self.k ))

        weight = None
        wvar = None

        for j in range(self.d):
            # Inverting V
            Vj = np.linalg.inv(V[j])
            wj = omega[j]
            yj = y[j]

            if np.abs(np.sum(yj)) > 0:

                V11= yj[0]*Vj[0,0] + yj[1]*Vj[1,1]
                V22= yj[0]*Vj[1,1] + yj[1]*Vj[0,0]
                Vb= (Vj[0,1]+Vj[1,0])/2.
                w1= yj[0]*wj[0] + yj[1]*wj[1]
                w2= yj[0]*wj[1] + yj[1]*wj[0]

                alpha12 = w1*np.sqrt(V11- (Vb**2)/V22)
                alpha21 = w2*np.sqrt(V22- (Vb**2)/V11)
                Sv = V11 + V22 + 2*Vb
                beta = (w1*(V11+Vb) + w2*(V22+Vb))/np.sqrt(Sv)
                Omegasqr = V11*(w1**2) + V22*(w2**2) + 2*Vb*w1*w2
                gamma12 = (V11*w1 + Vb*w2)/np.sqrt(V11)
                sigma12 = (Vb**2 - 2*V11*V22 - V22*Vb)/(Vb**2)
                v = V11*V22/Vb
                sp = np.sqrt(2./np.pi)
                expoOmegaBeta = np.exp(-0.5*(Omegasqr- beta**2))
                expoB = np.exp(-0.5*(beta**2))

                # Integrals
                params = np.array([V22, Vb, w1, w2, alpha12])
                I0 = nquad(I0_10, [[-alpha12,self.infinity]],params, opts=[{},{'weight': weight},{'wvar': wvar},{}])[0]
                I1 = nquad(I1_10, [[-alpha12,self.infinity]],params, opts=[{},{'weight': weight},{'wvar': wvar},{}])[0]
                I2 = nquad(I2_10, [[-alpha12,self.infinity]],params, opts=[{},{'weight': weight},{'wvar': wvar},{}])[0]
                Phi = 1. - norm.cdf(-beta)
                # Normalization
                Z = (w1/alpha12)*I0
                # Mean
                f1 =  -sp*(Vb/np.sqrt(V22*Sv))*expoOmegaBeta*Phi + I1
                f2 = -sp*np.sqrt(V22/Sv)*expoOmegaBeta*Phi
                f_ = np.array([f1,f2]) /Z
                f[j] = yj[0]*f_ + yj[1]*np.flip(f_)
                # Variance
                aux11 = beta - np.sqrt(Sv)*( (1.+V22/(Vb*sigma12))*w1 - V22*w2/(Vb*sigma12))
                aux22 = beta - np.sqrt(Sv)*(Vb*w1 + V22*w2)/(Vb+V22)
                aux12 =  (V11*V22 - (Vb**2))/(V11+Vb)
                df11 = - Z*(alpha12/w1)**2 + (alpha12/w1)*I2 + sp*((Vb**3)/(Sv*np.sqrt(V22**3)))*sigma12*expoOmegaBeta*(expoB/np.sqrt(2*np.pi) + aux11*Phi)
                df22 = -sp*np.sqrt(V22)*((Vb+V22)/Sv)*expoOmegaBeta*(expoB/np.sqrt(2*np.pi) + aux22*Phi)
                df12 = -sp*np.sqrt(V22)*((Vb+V11)/Sv)*expoOmegaBeta*(expoB/np.sqrt(2*np.pi) - aux12*(w1-w2)*Phi/np.sqrt(Sv) )
                Df = np.array([[df11, df12], [df12, df22]]) / Z
                df[j] =  yj[0]*Df + yj[1]*np.flip(Df) - np.outer(f[j], f[j].T)

            else:
                V11= Vj[0,0]
                V22= Vj[1,1]
                Vb= (Vj[0,1]+Vj[1,0])/2.
                w1= wj[0]
                w2= wj[1]

                alpha12 = w1*np.sqrt(V11- (Vb**2)/V22)
                alpha21 = w2*np.sqrt(V22- (Vb**2)/V11)
                gamma12 = (V11*w1 + Vb*w2)/np.sqrt(V11)
                v = V11*V22/(Vb**2)
                sp = np.sqrt(2./np.pi)
                expoA = np.exp(-0.5*(alpha21**2))
                expoG = np.exp(-0.5*(gamma12**2))

                # Integrals
                params = np.array([V22, Vb, w1, w2, alpha12])
                I0 = nquad(I0_00, [[-self.infinity,-alpha12]],params, opts=[{},{'weight': weight},{'wvar': wvar},{}])[0]
                I1 = nquad(I1_00, [[-self.infinity,-alpha12]],params, opts=[{},{'weight': weight},{'wvar': wvar},{}])[0]
                I2 = nquad(I2_00, [[-self.infinity,-alpha12]],params, opts=[{},{'weight': weight},{'wvar': wvar},{}])[0]
                Phi = norm.cdf(-gamma12)
                # Normalization
                Z = (w1/alpha12)*I0
                # Mean
                f1 = - sp*(Vb/np.sqrt(V11*V22))*expoA*Phi + I1
                f2 = -sp*np.sqrt(V22/V11)*expoA*Phi
                f[j] = np.array([f1,f2]) / Z
                # Variance
                df11 = sp*((Vb**3)/(V11*np.sqrt(V22**3)))*expoA*(-expoG*(1.-2*v)/np.sqrt(2*np.pi) + Vb*w2*(1.-v)*Phi/np.sqrt(V11)) - Z*((alpha12/w1)**2) + (alpha12/w1)*I2
                df22 = sp*np.sqrt(V22)*(Vb/V11)*expoA*( expoG/np.sqrt(2*np.pi) +  np.sqrt(V11)*(alpha21**2)*Phi/(w2*Vb) )
                df12 = sp*np.sqrt(V22)*expoA*expoG/np.sqrt(2*np.pi)
                Df = np.array([[df11, df12], [df12, df22]]) / Z
                df[j] = Df - np.outer(f[j], f[j].T)

        return f, df


    def f_ch_argmax_k1(self, y, omega, V):
        """f_out and df_out for the argmax loss (k = 1)"""
        omega = omega.reshape(omega.shape[0])
        V = V.reshape(V.shape[0])
        y = 2.*y.reshape(y.shape[0]) - 1.
        f = np.zeros(omega.shape)
        df = np.zeros(V.shape)
        """f_out and df_out for the argmax loss (k = 1)"""
        for j in range(self.d):
            e_in = -y[j]*omega[j]/np.sqrt(2*V[j])
            Z = 0.5*erfc(e_in)
            norm = 1./np.sqrt(2*np.pi*V[j])
            f_ = y[j]*norm*np.exp(-e_in**2) / np.maximum(self.tol, Z)
            df_ = -omega[j]*f_/V[j] - f_*f_
            f[j] = f_
            df[j] = df_

        f = f.reshape((f.shape[0], 1))
        df = df.reshape((df.shape[0], 1, 1))

        return f, df


    def f_ch_probit(self, y, omega, V):
        """Compute g and g' for probit channel (k=1)"""
        v = V.reshape(V.shape[0])
        y = y.reshape(y.shape[0])
        y = 2*y  - 1.
        w = omega.reshape(omega.shape[0])
        phi = -y * w / np.sqrt(2 * (v + self.var_noise))
        g = 2 * y / (np.sqrt(2 * np.pi * (v + self.var_noise)) * erfcx(phi))
        dg = -g * (w / (v + self.var_noise) + g)

        g = g.reshape((g.shape[0], 1))
        dg = dg.reshape((dg.shape[0], 1, 1))

        return g, dg
