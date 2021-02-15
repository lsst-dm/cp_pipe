# This file is part of cp_pipe.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""Calculation of brighter-fatter effect correlations and kernels."""

__all__ = ['BrighterFatterKernelSolveTask',
           'BrighterFatterKernelSolveConfig']

import numpy as np

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from lsst.ip.isr import (BrighterFatterKernel)
from ._lookupStaticCalibration import lookupStaticCalibration


class BrighterFatterKernelSolveConnections(pipeBase.PipelineTaskConnections,
                                           dimensions=("instrument", "exposure", "detector")):
    dummy = cT.Input(
        name="raw",
        doc="Dummy exposure.",
        storageClass='Exposure',
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera associated with this data.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
        lookupFunction=lookupStaticCalibration,
    )
    inputPtc = cT.PrerequisiteInput(
        name="ptc",
        doc="Photon transfer curve dataset.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    outputBFK = cT.Output(
        name="brighterFatterKernel",
        doc="Output measured brighter-fatter kernel.",
        storageClass="BrighterFatterKernel",
        dimensions=("instrument", "detector"),
    )


class BrighterFatterKernelSolveConfig(pipeBase.PipelineTaskConfig,
                                      pipelineConnections=BrighterFatterKernelSolveConnections):
    level = pexConfig.ChoiceField(
        doc="The level at which to calculate the brighter-fatter kernels",
        dtype=str,
        default="AMP",
        allowed={
            "AMP": "Every amplifier treated separately",
            "DETECTOR": "One kernel per detector",
        }
    )
    ignoreAmpsForAveraging = pexConfig.ListField(
        dtype=str,
        doc="List of amp names to ignore when averaging the amplifier kernels into the detector"
        " kernel. Only relevant for level = DETECTOR",
        default=[]
    )
    xcorrCheckRejectLevel = pexConfig.Field(
        dtype=float,
        doc="Rejection level for the sum of the input cross-correlations. Arrays which "
        "sum to greater than this are discarded before the clipped mean is calculated.",
        default=2.0
    )
    nSigmaClip = pexConfig.Field(
        dtype=float,
        doc="Number of sigma to clip when calculating means for the cross-correlation",
        default=5
    )
    forceZeroSum = pexConfig.Field(
        dtype=bool,
        doc="Force the correlation matrix to have zero sum by adjusting the (0,0) value?",
        default=False,
    )
    useAmatrix = pexConfig.Field(
        dtype=bool,
        doc="Use the PTC 'a' matrix instead of the average of measured covariances?",
        default=False,
    )

    maxIterSuccessiveOverRelaxation = pexConfig.Field(
        dtype=int,
        doc="The maximum number of iterations allowed for the successive over-relaxation method",
        default=10000
    )
    eLevelSuccessiveOverRelaxation = pexConfig.Field(
        dtype=float,
        doc="The target residual error for the successive over-relaxation method",
        default=5.0e-14
    )

    # These are unused.  Are they worth implementing?
    correlationQuadraticFit = pexConfig.Field(
        dtype=bool,
        doc="Use a quadratic fit to find the correlations instead of simple averaging?",
        default=False,
    )
    correlationModelRadius = pexConfig.Field(
        dtype=int,
        doc="Build a model of the correlation coefficients for radii larger than this value in pixels?",
        default=100,
    )
    correlationModelSlope = pexConfig.Field(
        dtype=float,
        doc="Slope of the correlation model for radii larger than correlationModelRadius",
        default=-1.35,
    )


class BrighterFatterKernelSolveTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Measure appropriate Brighter-Fatter Kernel from the PTC dataset.

    """
    ConfigClass = BrighterFatterKernelSolveConfig
    _DefaultName = 'cpBfkMeasure'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.

        Parameters
        ----------
        butlerQC : `lsst.daf.butler.butlerQuantumContext.ButlerQuantumContext`
            Butler to operate on.
        inputRefs : `lsst.pipe.base.connections.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `lsst.pipe.base.connections.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)

        # Use the dimensions to set calib/provenance information.
        inputs['inputDims'] = inputRefs.inputPtc.dataId.byName()

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputPtc, dummy, camera, inputDims):
        """Combine covariance information from PTC into brighter-fatter kernels.

        Parameters
        ----------
        inputPtc : `lsst.ip.isr.PhotonTransferCurveDataset`
            PTC data containing per-amplifier covariance measurements.
        dummy : `lsst.afw.image.Exposure
            The exposure used to select the appropriate PTC dataset.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera to use for camera geometry information.
        inputDims : `lsst.daf.butler.DataCoordinate` or `dict`
            DataIds to use to populate the output calibration.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The resulst struct containing:

            ``outputBfk`` : `lsst.ip.isr.BrighterFatterKernel`
                Resulting Brighter-Fatter Kernel.
        """
        if len(dummy) == 0:
            self.log.warn("No dummy exposure found.")

        detector = camera[inputDims['detector']]
        detName = detector.getName()

        if self.config.level == 'DETECTOR':
            detectorCorrList = list()

        bfk = BrighterFatterKernel(camera=camera, detectorId=detector.getId(), level=self.config.level)
        bfk.means = inputPtc.finalMeans  # ADU
        bfk.rawMeans = inputPtc.rawMeans  # ADU

        bfk.variances = inputPtc.finalVars  # ADU^2
        bfk.rawXcorrs = inputPtc.covariances  # ADU^2

        bfk.gain = inputPtc.gain
        bfk.noise = inputPtc.noise
        bfk.meanXCorrs = dict()

        for amp in detector:
            ampName = amp.getName()
            mask = inputPtc.expIdMask[ampName]

            gain = bfk.gain[ampName]
            fluxes = np.array(bfk.means[ampName])[mask]
            variances = np.array(bfk.variances[ampName])[mask]
            xCorrList = [np.array(xcorr) for xcorr in bfk.rawXcorrs[ampName]]
            xCorrList = np.array(xCorrList)[mask]

            fluxes = np.array([flux*gain for flux in fluxes])  # Now in e^-
            variances = np.array([variance*gain*gain for variance in variances])  # Now in e^2-

            # This should duplicate the else block in generateKernel@L1358,
            # which in turn is based on Coulton et al Equation 22.
            scaledCorrList = list()
            for xcorrNum, (xcorr, flux, var) in enumerate(zip(xCorrList, fluxes, variances), 1):
                q = np.array(xcorr) * gain * gain  # xcorr now in e^-
                q *= 2.0  # Remove factor of 1/2 applied in PTC.
                self.log.info("Amp: %s %d/%d Flux: %f  Var: %f  Q(0,0): %g  Q(1,0): %g  Q(0,1): %g",
                              ampName, xcorrNum, len(xCorrList), flux, var, q[0][0], q[1][0], q[0][1])

                # Normalize by the flux, which removes the (0,0)
                # component attributable to Poisson noise.
                q[0][0] -= 2.0*(flux)

                if q[0][0] > 0.0:
                    self.log.warn("Amp: %s %d skipped due to value of (variance-mean)=%f",
                                  ampName, xcorrNum, q[0][0])
                    continue

                q /= -2.0*(flux**2)
                scaled = self._tileArray(q)

                xcorrCheck = np.abs(np.sum(scaled))/np.sum(np.abs(scaled))
                if xcorrCheck > self.config.xcorrCheckRejectLevel:
                    self.log.warn("Amp: %s %d skipped due to value of triangle-inequality sum %f",
                                  ampName, xcorrNum, xcorrCheck)
                    continue

                scaledCorrList.append(scaled)
                self.log.info("Amp: %s %d/%d  Final: %g  XcorrCheck: %f",
                              ampName, xcorrNum, len(xCorrList), q[0][0], xcorrCheck)

            if len(scaledCorrList) == 0:
                self.log.warn("Amp: %s All inputs rejected for amp!", ampName)
                bfk.ampKernels[ampName] = np.zeros_like(np.pad(scaled, ((1, 1))))
                continue

            if self.config.level == 'DETECTOR':
                detectorCorrList.extend(scaledCorrList)

            if self.config.useAmatrix:
                # This is mildly wasteful
                preKernel = np.pad(self._tileArray(np.array(inputPtc.aMatrix[ampName])), ((1, 1)))
            else:
                preKernel = self.averageCorrelations(scaledCorrList, f"Amp: {ampName}")

            finalSum = np.sum(preKernel)
            center = int((preKernel.shape[0] - 1) / 2)
            bfk.meanXCorrs[ampName] = preKernel

            postKernel = self.successiveOverRelax(preKernel)
            bfk.ampKernels[ampName] = postKernel
            self.log.info("Amp: %s Sum: %g  Center Info Pre: %g  Post: %g",
                          ampName, finalSum, preKernel[center, center], postKernel[center, center])

        # Assemble a detector kernel?
        if self.config.level == 'DETECTOR':
            preKernel = self.averageCorrelations(detectorCorrList, f"Det: {detName}")
            finalSum = np.sum(preKernel)
            center = int((preKernel.shape[0] - 1) / 2)

            postKernel = self.successiveOverRelax(preKernel)
            bfk.detKernels[detName] = postKernel
            self.log.info("Det: %s Sum: %g  Center Info Pre: %g  Post: %g",
                          detName, finalSum, preKernel[center, center], postKernel[center, center])

        bfk.shape = postKernel.shape

        return pipeBase.Struct(
            outputBFK=bfk,
        )

    def averageCorrelations(self, xCorrList, name):
        """Average input correlations.

        Parameters
        ----------
        xCorrList : `list` [`numpy.array`]
            List of cross-correlations.
        name : `str`
            Name for log messages.

        Returns
        -------
        meanXcorr : `numpy.array`
            The averaged cross-correlation.
        """
        meanXcorr = np.zeros_like(xCorrList[0])
        xCorrList = np.transpose(xCorrList)
        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(self.config.nSigmaClip)
        for i in range(np.shape(meanXcorr)[0]):
            for j in range(np.shape(meanXcorr)[1]):
                meanXcorr[i, j] = afwMath.makeStatistics(xCorrList[i, j],
                                                         afwMath.MEANCLIP, sctrl).getValue()

        # To match previous definitions, pad by one element.
        meanXcorr = np.pad(meanXcorr, ((1, 1)))
        center = int((meanXcorr.shape[0] - 1) / 2)
        if self.config.forceZeroSum or True:
            totalSum = np.sum(meanXcorr)
            meanXcorr[center, center] -= totalSum
            self.log.info("%s Zero-Sum Scale: %g", name, totalSum)

        return meanXcorr

    @staticmethod
    def _tileArray(in_array):
        """Given an input quarter-image, tile/mirror it and return full image.

        Given a square input of side-length n, of the form

        input = array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

        return an array of size 2n-1 as

        output = array([[ 9,  8,  7,  8,  9],
                        [ 6,  5,  4,  5,  6],
                        [ 3,  2,  1,  2,  3],
                        [ 6,  5,  4,  5,  6],
                        [ 9,  8,  7,  8,  9]])

        Parameters:
        -----------
        input : `np.array`
            The square input quarter-array

        Returns:
        --------
        output : `np.array`
            The full, tiled array
        """
        assert(in_array.shape[0] == in_array.shape[1])
        length = in_array.shape[0] - 1
        output = np.zeros((2*length + 1, 2*length + 1))

        for i in range(length + 1):
            for j in range(length + 1):
                output[i + length, j + length] = in_array[i, j]
                output[-i + length, j + length] = in_array[i, j]
                output[i + length, -j + length] = in_array[i, j]
                output[-i + length, -j + length] = in_array[i, j]
        return output

    def successiveOverRelax(self, source, maxIter=None, eLevel=None):
        """An implementation of the successive over relaxation (SOR) method.

        A numerical method for solving a system of linear equations
        with faster convergence than the Gauss-Seidel method.

        Parameters:
        -----------
        source : `numpy.ndarray`
            The input array.
        maxIter : `int`, optional
            Maximum number of iterations to attempt before aborting.
        eLevel : `float`, optional
            The target error level at which we deem convergence to have
            occurred.

        Returns:
        --------
        output : `numpy.ndarray`
            The solution.
        """
        if not maxIter:
            maxIter = self.config.maxIterSuccessiveOverRelaxation
        if not eLevel:
            eLevel = self.config.eLevelSuccessiveOverRelaxation

        assert source.shape[0] == source.shape[1], "Input array must be square"
        # initialize, and set boundary conditions
        func = np.zeros([source.shape[0] + 2, source.shape[1] + 2])
        resid = np.zeros([source.shape[0] + 2, source.shape[1] + 2])
        rhoSpe = np.cos(np.pi/source.shape[0])  # Here a square grid is assumed

        # Calculate the initial error
        for i in range(1, func.shape[0] - 1):
            for j in range(1, func.shape[1] - 1):
                resid[i, j] = (func[i, j - 1] + func[i, j + 1] + func[i - 1, j]
                               + func[i + 1, j] - 4*func[i, j] - source[i - 1, j - 1])
        inError = np.sum(np.abs(resid))

        # Iterate until convergence
        # We perform two sweeps per cycle,
        # updating 'odd' and 'even' points separately
        nIter = 0
        omega = 1.0
        dx = 1.0
        while nIter < maxIter*2:
            outError = 0
            if nIter%2 == 0:
                for i in range(1, func.shape[0] - 1, 2):
                    for j in range(1, func.shape[1] - 1, 2):
                        resid[i, j] = float(func[i, j-1] + func[i, j + 1] + func[i - 1, j]
                                            + func[i + 1, j] - 4.0*func[i, j] - dx*dx*source[i - 1, j - 1])
                        func[i, j] += omega*resid[i, j]*.25
                for i in range(2, func.shape[0] - 1, 2):
                    for j in range(2, func.shape[1] - 1, 2):
                        resid[i, j] = float(func[i, j - 1] + func[i, j + 1] + func[i - 1, j]
                                            + func[i + 1, j] - 4.0*func[i, j] - dx*dx*source[i - 1, j - 1])
                        func[i, j] += omega*resid[i, j]*.25
            else:
                for i in range(1, func.shape[0] - 1, 2):
                    for j in range(2, func.shape[1] - 1, 2):
                        resid[i, j] = float(func[i, j - 1] + func[i, j + 1] + func[i - 1, j]
                                            + func[i + 1, j] - 4.0*func[i, j] - dx*dx*source[i - 1, j - 1])
                        func[i, j] += omega*resid[i, j]*.25
                for i in range(2, func.shape[0] - 1, 2):
                    for j in range(1, func.shape[1] - 1, 2):
                        resid[i, j] = float(func[i, j - 1] + func[i, j + 1] + func[i - 1, j]
                                            + func[i + 1, j] - 4.0*func[i, j] - dx*dx*source[i - 1, j - 1])
                        func[i, j] += omega*resid[i, j]*.25
            outError = np.sum(np.abs(resid))
            if outError < inError*eLevel:
                break
            if nIter == 0:
                omega = 1.0/(1 - rhoSpe*rhoSpe/2.0)
            else:
                omega = 1.0/(1 - rhoSpe*rhoSpe*omega/4.0)
            nIter += 1

        if nIter >= maxIter*2:
            self.log.warn("Failure: SuccessiveOverRelaxation did not converge in %s iterations."
                          "\noutError: %s, inError: %s," % (nIter//2, outError, inError*eLevel))
        else:
            self.log.info("Success: SuccessiveOverRelaxation converged in %s iterations."
                          "\noutError: %s, inError: %s", nIter//2, outError, inError*eLevel)
        return func[1: -1, 1: -1]
