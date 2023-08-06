import numpy as np
import matplotlib.pyplot as plt


class EnsembleSquareRootSmoother:
    def __init__(self, eakf):
        """
        eakf: EnsembleAdjustmentKalmanFilter
        """
        self.eakf = eakf

    def ensrf(x, y, z, oev, uhh=True, H=np.array([0, 0, 0, 1])):
        """
        Args:
            x: latent states or parameters
            y: Hx
            z: observation
            oev: observational error variance
            H: observation operator
        """
        x = np.array(x)
        y = np.array(y)

        xbar_hat = np.mean(x, axis=1)
        ybar_hat = np.mean(y)

        Pb = np.cov(x)
        Pyy = np.cov(y)
        if uhh:
            K = Pb @ H.T / (Pyy + oev)  # eq 2
        else:
            Pb = np.cov(x, y)
            H = np.array([0, 0, 1])
            K = Pb @ H.T / (Pyy + oev)  # eq 2
            K = K[:-1]
        xbar_a = xbar_hat + K * (np.mean(z) - ybar_hat)  # eq 4
        Ktilde = 1 / (1 + np.sqrt(oev / (Pyy + oev))) * K  # eq 7
        Ktilde = np.array([Ktilde]).T
        inn = np.array([ybar_hat - y])
        xpost = x - np.array([xbar_hat]).T + np.array([xbar_a]).T +\
            Ktilde @ inn  # eq 5

        return xpost

    def smooth(self, num_steps, n_real, plot=False):
        lag = 1
        x_list = np.array(self.eakf.x_list)
        xhat_list = np.array(self.eakf.xhat_list)
        θ_list = np.array(self.eakf.θ_list)

        x_lag_list = []
        θ_lag_list = []

        for k in range(self.eakf.data.n_t - 1):
            z = self.eakf.data.i[k + lag]
            oev = self.eakf.oev(z)
            H = np.array([0, 0, 0, 1])

            # Smooth STATES
            x = x_list[k, :, :]
            xhat = xhat_list[k + lag, :, :]

            y = H @ xhat
            ybar_hat = np.mean(y)
            Pyy = np.cov(y)
            Pba = np.cov(x, xhat)[4:, 4:]
            # eq 9
            K_lag = (H @ Pba).T / (Pyy + oev)

            x_bar = np.mean(x, axis=1)
            xhat_bar = np.array([np.mean(xhat, axis=1)])
            # eq 8
            xbar_lag = x_bar + K_lag * (z - ybar_hat)
            # eq 15
            Ktilde_lag = K_lag / (1.0 + np.sqrt(oev / (Pyy + oev)))
            # eq 13
            x_lag = (
                (x - np.array([x_bar]).T)
                + np.array([xbar_lag]).T
                - Ktilde_lag * H @ (xhat - xhat_bar.T)
            )

            x_lag = np.clip(x_lag, 0, self.eakf.data.N)
            x_lag = self.eakf.SIR(*x_lag)
            x_lag_list.append(x_lag)

            # Smooth PARAMS
            θ = θ_list[k, :, :]
            H = np.array([0, 0, 1])
            Pba = np.cov(θ, y)

            # eq 9
            K_lag = (H @ Pba).T / (Pyy + oev)
            K_lag = K_lag[:-1]

            θ_bar = np.array([np.mean(θ, axis=1)]).T
            # eq 8
            θ_bar_lag = θ_bar + np.array([K_lag * (z - ybar_hat)]).T
            # eq 15
            Ktilde_lag = K_lag / (1.0 + np.sqrt(oev / (Pyy + oev)))
            # eq 13
            θ_lag = (
                (θ - θ_bar)
                + θ_bar_lag
                - np.array([Ktilde_lag]).T @ np.array([y - ybar_hat.T])
            )

            θ_lag = np.clip(θ_lag, 0, 10)
            θ_lag = self.eakf.ModelParams(*θ_lag)
            θ_lag_list.append(θ_lag)

        # plot
        if plot:
            x_lag_list = np.array(x_lag_list)
            x_lag_means = np.mean(x_lag_list, axis=2)

            fig, ax = plt.subplots(3)
            ax[0].plot(x_lag_list[:, 0, :], color="gray", alpha=0.1)
            ax[0].plot(x_lag_means[:, 0], color="black")
            ax[0].plot(self.eakf.data.S, ".")

            ax[1].plot(x_lag_list[:, 1, :], color="gray", alpha=0.1)
            ax[1].plot(x_lag_means[:, 1], color="black")
            ax[1].plot(self.eakf.data.I, ".")

            ax[2].plot(x_lag_list[:, 2, :], color="gray", alpha=0.1)
            ax[2].plot(x_lag_means[:, 2], color="black")
            ax[2].plot(self.eakf.data.R, ".")

        θ_lag_list = np.array(θ_lag_list)

        if plot:
            fig, ax = plt.subplots(2)

            ax[0].plot(x_lag_list[:, 3, :], color="gray", alpha=0.1)
            ax[0].plot(self.eakf.data.i, ".")
            ax[0].plot(x_lag_means[:, 3], color="black")
            ax[0].plot(np.mean(x_list[:, 3, :], axis=1), color="green")

            ax[1].plot(θ_lag_list[:, 0, :], color="gray", alpha=0.1)
            ax[1].plot(self.eakf.data.beta, color="red")
            ax[1].plot(np.mean(θ_lag_list[:, 0, :], axis=1), color="black")
            # ax[1].plot(np.mean(θ_list[:,0,:], axis=1)[:-1], color="green")
            ax[1].set_xlabel("day")
            ax[1].set_ylabel(r"$\beta(t)$")

            fig.suptitle(f"EnSRS window size {lag} with adaptive inflation")

        for lag in np.arange(2, 10):
            θ_list = np.array(θ_lag_list).copy()
            x_list = np.array(x_lag_list).copy()

            x_lag_list = []
            θ_lag_list = []

            for k in range(self.eakf.data.n_t - lag):
                z = self.eakf.data.i[k + lag]
                oev = np.maximum(10, z**2 / 50)
                H = np.array([0, 0, 0, 1])

                # Smooth STATES
                x = x_list[k, :, :]
                xhat = xhat_list[k + lag, :, :]

                y = H @ xhat
                ybar_hat = np.mean(y)
                Pyy = np.cov(y)
                Pba = np.cov(x, xhat)[4:, 4:]
                # eq 9
                K_lag = (H @ Pba).T / (Pyy + oev)

                x_bar = np.mean(x, axis=1)
                xhat_bar = np.array([np.mean(xhat, axis=1)])
                # eq 8
                xbar_lag = x_bar + K_lag * (z - ybar_hat)
                # eq 15
                Ktilde_lag = K_lag / (1.0 + np.sqrt(oev / (Pyy + oev)))
                # eq 13
                x_lag = (
                    (x - np.array([x_bar]).T)
                    + np.array([xbar_lag]).T
                    - Ktilde_lag * H @ (xhat - xhat_bar.T)
                )

                x_lag = np.clip(x_lag, 0, self.eakf.data.N)
                x_lag = self.eakf.SIR(*x_lag)
                x_lag_list.append(x_lag)

                # Smooth PARAMS
                θ = θ_list[k, :, :]
                H = np.array([0, 0, 1])
                Pba = np.cov(θ, y)

                # eq 9
                K_lag = (H @ Pba).T / (Pyy + oev)
                K_lag = K_lag[:-1]

                θ_bar = np.array([np.mean(θ, axis=1)]).T
                # eq 8
                θ_bar_lag = θ_bar + np.array([K_lag * (z - ybar_hat)]).T
                # eq 15
                Ktilde_lag = K_lag / (1.0 + np.sqrt(oev / (Pyy + oev)))
                # eq 13
                θ_lag = (
                    (θ - θ_bar) + θ_bar_lag
                    - np.array([Ktilde_lag]).T @ np.array([y - ybar_hat.T])
                )

                θ_lag = np.clip(θ_lag, 0, 10)
                θ_lag = self.eakf.ModelParams(*θ_lag)
                θ_lag_list.append(θ_lag)

            # plot
            if plot:
                x_lag_list = np.array(x_lag_list)
                x_lag_means = np.mean(x_lag_list, axis=2)

                fig, ax = plt.subplots(3)
                ax[0].plot(x_lag_list[:, 0, :], color="gray", alpha=0.1)
                ax[0].plot(x_lag_means[:, 0], color="black")
                ax[0].plot(self.eakf.data.S, ".")

                ax[1].plot(x_lag_list[:, 1, :], color="gray", alpha=0.1)
                ax[1].plot(x_lag_means[:, 1], color="black")
                ax[1].plot(self.eakf.data.I, ".")

                ax[2].plot(x_lag_list[:, 2, :], color="gray", alpha=0.1)
                ax[2].plot(x_lag_means[:, 2], color="black")
                ax[2].plot(self.eakf.data.R, ".")

            θ_lag_list = np.array(θ_lag_list)

            if plot:
                fig, ax = plt.subplots(2)

                ax[0].plot(x_lag_list[:, 3, :], color="gray", alpha=0.1)
                ax[0].plot(self.eakf.data.i, ".")
                ax[0].plot(x_lag_means[:, 3], color="black")
                # ax[0].plot(np.mean(x_list[:,3,:], axis=1), color='green')

                ax[1].plot(θ_lag_list[:, 0, :], color="gray", alpha=0.1)
                ax[1].plot(self.eakf.data.beta, color="red")
                ax[1].plot(np.mean(θ_lag_list[:, 0, :], axis=1), color="black")
                ax[1].set_xlabel("day")
                ax[1].set_ylabel(r"$\beta(t)$")

                fig.suptitle(f"EnSRS window size {lag}")
