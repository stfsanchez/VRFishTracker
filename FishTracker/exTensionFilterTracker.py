


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: © 2022 Boris Lenseigne, Renaud Bastien, Stephane Sanchez <stephane.sanchez@ut-capitole.fr>
Université Toulouse Capitole - Institut de Recherche en Informatique de Toulouse
Real Expression Artificial Life (REVA) Research Team
https://www.irit.fr/departement/calcul-intensif-simulation-optimisation/reva/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm  # Matrix square root
from scipy.linalg import eigh
from scipy.linalg import cholesky
# import matplotlib as mpl
# from matplotlib import cm
# import matplotlib.transforms as transforms
# from mpl_toolkits.mplot3d import Axes3D

#https://stackoverflow.com/questions/71232192/efficient-matrix-square-root-of-large-symmetric-positive-semidefinite-matrix-in
def sqrtm3(B):
    return cholesky(B)

def sqrtm2(B):
    D, V = eigh(B)
    return (V * np.sqrt(D)) @ V.T

def sqrtm3(B):
    return sqrtm(B)

class ExtensionFilter:
    """
        A class for the extension filter adpated from:
        Tracking of Extended Objects and Group Targets
        using Random Matrices – A New Approach, Michael Feldmann,
        Dietrich Fränken IEEE Transactions on Signal Processing, 2008

        The filter estimates the position and speed of a target in n-dimensions
        It assumes a constant velocity process model by default.
        This can be changed on the fly
    """

    def __init__(self,  Qscale, Rscale, x0=None, nDims=3, fps=1):
        """
        Initialize an ExtensionFilter Instance.

        Parameters
        ----------
        Qscale : float
            Scale factor for the process noise covariance. Unless set
            explicitely, the process noise covariance is:
                Qscale*np.identity(2*nDims)
            It can be set manually after instanciation, if you want something
            completely different.
        Rscale : float
            Scale factor for the measurement noise covariance. Unless set
            explicitely, the measurement noise covariance is:
                Qscale*np.identity(nDims)
            It can be set manually after instanciation, if you want something
            completely different.
        x0 : nDims x 1 array
            Initial state vector at startup. If ommited is set to zero for
            position and speed.
        nDims : int
            Number of dimensions of the cartesian space where the target is
            moving (typically 2 or 3 but this is no limitation.) Defaults to 3.
        fps : int
            Frames per second for the input stream, defaults to 1.
            It does not have to reflect the real fps, but you can play with it
            to tweak he filter's behaviour. Be carefull with it as it may also
            break it.
        Returns
        -------
        """
        self.nDims = nDims
        self.dt = 1 / fps
        ##############
        # Generate the necessary matrices based on nDims
        #############
        # 1- Process model : constant speed
        F = np.zeros((2 * self.nDims, 2 * self.nDims))
        for i in range(0, self.nDims):
            F[i, i] = 1
            F[i, self.nDims+i] = self.dt
        # Constant value for z
        #if (self.nDims > 2):
        #    F[2, 5] = 0.0
        for i in range(self.nDims, 2 * self.nDims):
            F[i, i] = 1
        self.F = F
        # 2-Measurement model
        H = np.zeros((self.nDims, 2 * self.nDims))
        for i in range(0, self.nDims):
            H[i, i] = 1
        self.H = H
        # 3-Process model noise covariance matrix:
        self.Q = Qscale * np.identity(2 * self.nDims)
        # 4-Sensor noise covariance matrix
        self.R = Rscale * np.identity(self.nDims)
        #self.R[2,2] = 10000*Rscale #favoriser le modèle prédictif
        # 5-Estimation error covariance matrix
        self.P_kk = np.identity(2 * self.nDims)
        self.P_kkm = np.zeros(self.P_kk.shape)
        # 6-State vector
        if x0 is not None:
            self.x_kk = x0
        else:
            self.x_kk = np.zeros((2 * self.nDims, 1))
        self.x_kkm = np.zeros(self.x_kk.shape)

        # Filter parameters :
        self.alpha_kk = 1
        # Filter reactivity (higher for a less agile target)
        self.Tau = 10
        # Innovation covariance scale factor
        self.z = 1
        # Target extension covariance matrix
        self.X_kk = np.identity(self.nDims)
        self.X_kkm = np.zeros(self.X_kk.shape)
        # A counter to track how many rounds have accured without measurement
        # updates
        self.predictionOnlyTurns = 0
        # Set to True if the extension update generated some complex values
        # during last iteration.
        # (it usually means the filter should be reset.)
        self.hadComplexValues = False

    def timeUpdate(self):
        """
        Temporal prediction step.

        Update the state vector and estimation covariance by application of
        the process model.

        Returns
        -------
        None.

        """
        self.x_kkm = np.matmul(self.F, self.x_kk)
        # Prediction covariance
        self.P_kkm = np.matmul(self.F,
                               np.matmul(self.P_kk,
                                         np.transpose(self.F))
                               ) + self.Q
        self.alpha_kkm = np.exp(-self.dt / self.Tau) * self.alpha_kk
        self.X_kkm = self.X_kk

    def measurementUpdate(self, yBar, YBar, nk):
        """
        Update step.

        Measurement update step, based on the measurement spread described by
        its center and covariance matrix.

        Parameters
        ----------
        yBar : nDimsx1 array
            Barycentrum of the measurement point cloud
        YBar : nDims x nDims array
            Covariance matrix of the measurement point cloud
        nk : int
            Number of points in the measurement
        Returns
        -------
        """
        # Reset flags
        self.b_hadComplexValues = False
        # Innovation covariance
        Y = self.z * self.X_kkm + self.R
        S = np.matmul(self.H,
                      np.matmul(self.P_kkm,
                                np.transpose(self.H))
                      ) + Y / nk
        # Kalman gain
        inv_S = np.linalg.inv(S)
        #if np.any(np.iscomplex(inv_S)):
        #    print("line 165 ... complex values")
        K = np.matmul(self.P_kkm,
                      np.matmul(np.transpose(self.H),
                                inv_S))
        # Update the state estimate and estimate covariance matrix as usual
        Hx_kkm = np.matmul(self.H, self.x_kkm)
        self.x_kk = (self.x_kkm +
                     np.matmul(K,
                               (yBar -
                                Hx_kkm)))
        self.P_kk = self.P_kkm - np.matmul(K,
                                           np.matmul(S,
                                                     np.transpose(K)))
        #################
        # Extension update
        #################

        # Measuremement error covariance matrix
        N_kkm = np.matmul((yBar - Hx_kkm),
                          np.transpose(yBar - Hx_kkm))

        # Compute the matrix square root of X, S and Y
        # Note : sqrtm: seems to be giving complex answers where it shouldn't...
        # (see: https://stackoverflow.com/questions/64424587/how-to-suppress-complex-numbers-in-the-calculation-of-the-square-root-of-a-posit
        #  https://github.com/scipy/scipy/pull/3556).
        # Until I got a better idea I just force it to be real and raise a flag
        # when it happens.
        sqrX = np.real(sqrtm3(self.X_kkm))
        #if np.any(np.iscomplex(sqrX)):
        #    print("line 194 ... complex values")
        #    sqrX = np.real(sqrX)
        #    self.b_hadComplexValues = True
        
        sqrS = np.real(sqrtm3(S))
        #if np.any(np.iscomplex(sqrS)):
        #    print("line 200 ... complex values")
        #    sqrS = np.real(sqrS)
        #    self.b_hadComplexValues = True
        
        sqrSinv = np.linalg.inv(sqrS)
        #if np.any(np.iscomplex(sqrSinv)):
        #    print("line 206 ... complex values")
        
        sqrY = np.real(sqrtm3(Y))
        #if np.any(np.iscomplex(sqrY)):
        #    print("line 210 ... complex values")
        #    sqrY = np.real(sqrY)
        #    self.b_hadComplexValues = True
        
        sqrYinv = np.linalg.inv(sqrY)
        #if np.any(np.iscomplex(sqrYinv)):
        #    print("line 216 ... complex values")

        Nhat_kkm = np.matmul(sqrX,
                             np.matmul(sqrSinv,
                                       np.matmul(N_kkm,
                                                 np.matmul(np.transpose(sqrSinv),
                                                           np.transpose(sqrX)
                                                           ))))

        Yhat_kkm = np.matmul(sqrX,
                             np.matmul(sqrYinv,
                                       np.matmul(YBar,
                                                 np.matmul(np.transpose(sqrYinv),
                                                           np.transpose(sqrX)
                                                           ))))
        self.alpha_kk = self.alpha_kkm + nk

        self.X_kk = ((1/self.alpha_kk) *
                     (self.alpha_kkm * self.X_kkm + Nhat_kkm + Yhat_kkm))
        self.predictionOnlyTurns = 0

    def propagatePrediction(self):
        """
        Propagate the predicted state to the corrected state in cases where no
        measurements are available

        Returns
        -------
        None.

        """
        self.x_kk = self.x_kkm
        self.X_kk = self.X_kkm
        self.P_kk = self.P_kkm
        self.alpha_kk = self.alpha_kkm
        self.predictionOnlyTurns += 1


def draw2DCovMatEllipse(ax, covMat, pos=(0, 0),  nstd=3.0, edgeColor='red'):
    """
    Create a plot of the covariance confidence ellipse from a covariance Matrix
    Stolen and adapted from:
    https://stats.stackexchange.com/questions/361017/proper-way-of-estimating-the-covariance-error-ellipse-in-2d/361334

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    covMat : 2x2 array
        Covariance Matrix
    pos : tuple
        The center of the matrix
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    edgeColor: see matplotlib documentation
        The color to draw with
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    """

    # Eigenvalues and eigenvectors of the covariance matrix.
    vals, vecs = np.linalg.eigh(covMat)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    # Ellipse
    ellipse1 = Ellipse(xy=pos, width=width, height=height, angle=theta,
                       edgecolor=edgeColor, fc='None', lw=2, zorder=4)
    ax.add_patch(ellipse1)
    plt.show()


def getExtendedMeasurement(Xtrue, timeStep, targetCovMat, npt=5):
    """
    Get a point cloud of measurements following the desired distribution

    Parameters
    ----------
    Xtrue : nx2 array
        Real [[x],[y]] position of the target
    timeStep : int
        Current simulation time step
    targetCovMat : 2x2 array
        Target Extention Covariance Matrix
    npt : int
        number of points to draw from the distribution (default = 5)

    Returns
    -------
    meas : nptx2 array
        Point cloud [[x],[y]] estimations of the position of the target
    """

    Xcurr = Xtrue[timeStep, :]
    meas = np.random.multivariate_normal(mean=(Xcurr[0], Xcurr[1]),
                                         cov=targetCovMat,
                                         size=npt)
    return meas


if __name__ == '__main__':
    # dTime step
    dt = 1
    # Simulation duration
    nPts = 500
    # time counter
    t = 0
    # We generate a nice trajectory from a Lissajou curve.
    # These are the positions that are used to generate extended measurements
    timeSteps = np.arange(nPts)
    Xtrue = np.empty((nPts, 2))
    Xtrue[:, 0] = 100 * np.sin(3 * (timeSteps * 2 * np.pi)/nPts)
    Xtrue[:, 1] = 50 * np.sin(2 * (timeSteps * 2 * np.pi)/nPts)
    # Arbitrary sensor meas covariance
    XtgtCov = np.matrix([[2, 0.5], [0.5, 1]])
    # The figure where we show everything
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    t = 0
    Qscale = 0.5
    Rscale = 0.1
    x0 = np.array([[0], [0], [1], [0]])
    nDims = 2
    tracker = ExtensionFilter(Qscale, Rscale, None, nDims, fps=1)

    while t < nPts-1:
        print("t = ", t)
        tracker.timeUpdate()
        #################
        # Get a point cloud as a measurement
        #################

        meas = getExtendedMeasurement(Xtrue, t, XtgtCov)
        # Get the barycentrum of the point cloud
        yBar = np.transpose(np.array([np.mean(meas, axis=0)]))
        # Get the covariance matrix of the measurements
        YBar = np.cov(np.transpose(meas))
        # number of measurements
        nk = meas.shape[0]
        tracker.measurementUpdate(yBar, YBar, nk)
        x_kk = tracker.x_kk
        X_kk = tracker.X_kk
        if t == 0:
            ax.plot(meas[:, 0], meas[:, 1], '.k', label='Noisy measure')
            ax.plot(Xtrue[:, 0], Xtrue[:, 1], '-b', label='True position')
            ax.plot(x_kk[0], x_kk[1], '+g', label='Estimated position')

        else:
            ax.plot(meas[:, 0], meas[:, 1], '.k')
            ax.plot(Xtrue[:, 0], Xtrue[:, 1], '-b')
            ax.plot(x_kk[0], x_kk[1], '+g')
        draw2DCovMatEllipse(ax,
                            XtgtCov,
                            pos=(Xtrue[t, 0], Xtrue[t, 1]),
                            nstd=3.0,
                            edgeColor='k')
        draw2DCovMatEllipse(ax,
                            X_kk,
                            pos=(x_kk[0], x_kk[1]),
                            nstd=3.0,
                            edgeColor='r')
        plt.draw()
        print("Position [", x_kk[0], ", ", x_kk[1], "]")
        t = t + 1
    ax.legend()

#    plt.autoscale(enable=True)
