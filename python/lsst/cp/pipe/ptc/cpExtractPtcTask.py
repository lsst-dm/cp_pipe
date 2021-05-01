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
import numpy as np

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.cp.pipe.utils import (arrangeFlatsByExpTime, arrangeFlatsByExpId,
                                sigmaClipCorrection)

import lsst.pipe.base.connectionTypes as cT

from .astierCovPtcUtils import (CovFastFourierTransform, computeCovDirect)
from .astierCovPtcFit import makeCovArray

from lsst.ip.isr import PhotonTransferCurveDataset
from lsst.ip.isr import IsrTask

__all__ = ['PhotonTransferCurveExtractConfig', 'PhotonTransferCurveExtractTask']


class PhotonTransferCurveExtractConnections(pipeBase.PipelineTaskConnections,
                                            dimensions=("instrument", "detector")):

    inputExp = cT.Input(
        name="ptcInputExposurePairs",
        doc="Input post-ISR processed exposure pairs (flats) to"
            "measure covariances from.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=False,
    )

    outputCovariances = cT.Output(
        name="ptcCovariances",
        doc="Extracted flat (co)variances.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )


class PhotonTransferCurveExtractConfig(pipeBase.PipelineTaskConfig,
                                       pipelineConnections=PhotonTransferCurveExtractConnections):
    """Configuration for the measurement of covariances from flats.
    """
    matchByExposureId = pexConfig.Field(
        dtype=bool,
        doc="Should exposures by matched by ID rather than exposure time?",
        default=False,
    )
    maximumRangeCovariancesAstier = pexConfig.Field(
        dtype=int,
        doc="Maximum range of covariances as in Astier+19",
        default=8,
    )
    covAstierRealSpace = pexConfig.Field(
        dtype=bool,
        doc="Calculate covariances in real space or via FFT? (see appendix A of Astier+19).",
        default=False,
    )
    binSize = pexConfig.Field(
        dtype=int,
        doc="Bin the image by this factor in both dimensions.",
        default=1,
    )
    minMeanSignal = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc="Minimum values (inclusive) of mean signal (in ADU) above which to consider, per amp."
            " The same cut is applied to all amps if this dictionary is of the form"
            " {'ALL_AMPS': value}",
        default={'ALL_AMPS': 0.0},
    )
    maxMeanSignal = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc="Maximum values (inclusive) of mean signal (in ADU) below which to consider, per amp."
            " The same cut is applied to all amps if this dictionary is of the form"
            " {'ALL_AMPS': value}",
        default={'ALL_AMPS': 1e6},
    )
    maskNameList = pexConfig.ListField(
        dtype=str,
        doc="Mask list to exclude from statistics calculations.",
        default=['SUSPECT', 'BAD', 'NO_DATA', 'SAT'],
    )
    nSigmaClipPtc = pexConfig.Field(
        dtype=float,
        doc="Sigma cut for afwMath.StatisticsControl()",
        default=5.5,
    )
    nIterSigmaClipPtc = pexConfig.Field(
        dtype=int,
        doc="Number of sigma-clipping iterations for afwMath.StatisticsControl()",
        default=3,
    )
    minNumberGoodPixelsForCovariance = pexConfig.Field(
        dtype=int,
        doc="Minimum number of acceptable good pixels per amp to calculate the covariances (via FFT or"
            " direclty).",
        default=10000,
    )
    thresholdDiffAfwVarVsCov00 = pexConfig.Field(
        dtype=float,
        doc="If the absolute fractional differece between afwMath.VARIANCECLIP and Cov00 "
            "for a region of a difference image is greater than this threshold (percentage), "
            "a warning will be issued.",
        default=1.,
    )
    detectorMeasurementRegion = pexConfig.ChoiceField(
        dtype=str,
        doc="Region of each exposure where to perform the calculations (amplifier or full image).",
        default='AMP',
        allowed={
            "AMP": "Amplifier of the detector.",
            "FULL": "Full image."
        }
    )
    numEdgeSuspect = pexConfig.Field(
        dtype=int,
        doc="Number of edge pixels to be flagged as untrustworthy.",
        default=0,
    )
    edgeMaskLevel = pexConfig.ChoiceField(
        dtype=str,
        doc="Mask edge pixels in which coordinate frame: DETECTOR or AMP?",
        default="DETECTOR",
        allowed={
            'DETECTOR': 'Mask only the edges of the full detector.',
            'AMP': 'Mask edges of each amplifier.',
        },
    )


class PhotonTransferCurveExtractTask(pipeBase.PipelineTask,
                                     pipeBase.CmdLineTask):
    """Task to measure covariances from flat fields.
    This task receives as input a list of flat-field images
    (flats), and sorts these flats in pairs taken at the
    same time (if there's a different number of flats,
    those flats are discarded). The mean, variance, and
    covariances are measured from the difference of the flat
    pairs at a given time. The variance is calculated
    via afwMath, and the covariance via the methods in Astier+19
    (appendix A). In theory, var = covariance[0,0]. This should
    be validated, and in the future, we may decide to just keep
    one (covariance).

    The measured covariances at a particular time (along with
    other quantities such as the mean) are stored in a PTC dataset
    object (`PhotonTransferCurveDataset`), which gets partially
    filled. The number of partially-filled PTC dataset objects
    will be less than the number of input exposures, but gen3
    requires/assumes that the number of input dimensions matches
    bijectively the number of output dimensions. Therefore, a
    number of "dummy" PTC dataset are inserted in the output list
    that has the partially-filled PTC datasets with the covariances.
    This output list will be used as input of
    `PhotonTransferCurveSolveTask`, which will assemble the multiple
    `PhotonTransferCurveDataset`s into a single one in order to fit
    the measured covariances as a function of flux to a particular
    model.

    Astier+19: "The Shape of the Photon Transfer Curve of CCD
    sensors", arXiv:1905.08677.
    """
    ConfigClass = PhotonTransferCurveExtractConfig
    _DefaultName = 'cpPtcExtract'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.

        Parameters
        ----------
        butlerQC : `~lsst.daf.butler.butlerQuantumContext.ButlerQuantumContext`
            Butler to operate on.
        inputRefs : `~lsst.pipe.base.connections.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `~lsst.pipe.base.connections.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)
        # Dictionary, keyed by expTime, with flat exposures
        if self.config.matchByExposureId:
            inputs['inputExp'] = arrangeFlatsByExpId(inputs['inputExp'])
        else:
            inputs['inputExp'] = arrangeFlatsByExpTime(inputs['inputExp'])
        # Ids of input list of exposures
        inputs['inputDims'] = [expId.dataId['exposure'] for expId in inputRefs.inputExp]
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputExp, inputDims):
        """Measure covariances from difference of flat pairs

        Parameters
        ----------
        inputExp : `dict` [`float`,
                        (`~lsst.afw.image.exposure.exposure.ExposureF`,
                        `~lsst.afw.image.exposure.exposure.ExposureF`, ...,
                        `~lsst.afw.image.exposure.exposure.ExposureF`)]
            Dictionary that groups flat-field exposures that have the same
            exposure time (seconds).

        inputDims : `list`
            List of exposure IDs.
        """
        # inputExp.values() returns a view, which we turn into a list. We then
        # access the first exposure to get teh detector.
        detector = list(inputExp.values())[0][0].getDetector()
        detNum = detector.getId()
        amps = detector.getAmplifiers()
        ampNames = [amp.getName() for amp in amps]

        # Each amp may have a different  min and max ADU signal specified in the config.
        maxMeanSignalDict = {ampName: 1e6 for ampName in ampNames}
        minMeanSignalDict = {ampName: 0.0 for ampName in ampNames}
        for ampName in ampNames:
            if 'ALL_AMPS' in self.config.maxMeanSignal:
                maxMeanSignalDict[ampName] = self.config.maxMeanSignal['ALL_AMPS']
            elif ampName in self.config.maxMeanSignal:
                maxMeanSignalDict[ampName] = self.config.maxMeanSignal[ampName]

            if 'ALL_AMPS' in self.config.minMeanSignal:
                minMeanSignalDict[ampName] = self.config.minMeanSignal['ALL_AMPS']
            elif ampName in self.config.minMeanSignal:
                minMeanSignalDict[ampName] = self.config.minMeanSignal[ampName]
        # These are the column names for `tupleRows` below.
        tags = [('mu', '<f8'), ('afwVar', '<f8'), ('i', '<i8'), ('j', '<i8'), ('var', '<f8'),
                ('cov', '<f8'), ('npix', '<i8'), ('ext', '<i8'), ('expTime', '<f8'), ('ampName', '<U3')]
        # Create a dummy ptcDataset
        dummyPtcDataset = PhotonTransferCurveDataset(ampNames, 'DUMMY',
                                                     self.config.maximumRangeCovariancesAstier)
        # Initialize amps of `dummyPtcDatset`.
        for ampName in ampNames:
            dummyPtcDataset.setAmpValues(ampName)
        # Output list with PTC datasets.
        partialPtcDatasetList = []
        # The number of output references needs to match that of input references:
        # initialize outputlist with dummy PTC datasets.
        for i in range(len(inputDims)):
            partialPtcDatasetList.append(dummyPtcDataset)

        if self.config.numEdgeSuspect > 0:
            isrTask = IsrTask()
            self.log.info(f"Masking {self.config.numEdgeSuspect} pixels from the edges "
                          "of all exposures as SUSPECT.")

        for expTime in inputExp:
            exposures = inputExp[expTime]
            if len(exposures) == 1:
                self.log.warn(f"Only one exposure found at expTime {expTime}. Dropping exposure "
                              f"{exposures[0].getInfo().getVisitInfo().getExposureId()}.")
                continue
            else:
                # Only use the first two exposures at expTime
                exp1, exp2 = exposures[0], exposures[1]
                if len(exposures) > 2:
                    self.log.warn(f"Already found 2 exposures at expTime {expTime}. "
                                  "Ignoring exposures: "
                                  f"{i.getInfo().getVisitInfo().getExposureId() for i in exposures[2:]}")
            # Mask pixels at the edge of the detector or of each amp
            if self.config.numEdgeSuspect > 0:
                isrTask.maskEdges(exp1, numEdgePixels=self.config.numEdgeSuspect,
                                  maskPlane="SUSPECT", level=self.config.edgeMaskLevel)
                isrTask.maskEdges(exp2, numEdgePixels=self.config.numEdgeSuspect,
                                  maskPlane="SUSPECT", level=self.config.edgeMaskLevel)
            expId1 = exp1.getInfo().getVisitInfo().getExposureId()
            expId2 = exp2.getInfo().getVisitInfo().getExposureId()
            nAmpsNan = 0
            partialPtcDataset = PhotonTransferCurveDataset(ampNames, '',
                                                           self.config.maximumRangeCovariancesAstier)
            for ampNumber, amp in enumerate(detector):
                ampName = amp.getName()
                # covAstier: [(i, j, var (cov[0,0]), cov, npix) for (i,j) in {maxLag, maxLag}^2]
                doRealSpace = self.config.covAstierRealSpace
                if self.config.detectorMeasurementRegion == 'AMP':
                    region = amp.getBBox()
                elif self.config.detectorMeasurementRegion == 'FULL':
                    region = None
                # `measureMeanVarCov` is the function that measures the variance and  covariances from
                # the difference image of two flats at the same exposure time.
                # The variable `covAstier` is of the form: [(i, j, var (cov[0,0]), cov, npix) for (i,j)
                # in {maxLag, maxLag}^2]
                muDiff, varDiff, covAstier = self.measureMeanVarCov(exp1, exp2, region=region,
                                                                    covAstierRealSpace=doRealSpace)
                # Correction factor for sigma clipping. Function returns 1/sqrt(varFactor),
                # so it needs to be squared. varDiff is calculated via afwMath.VARIANCECLIP.
                varFactor = sigmaClipCorrection(self.config.nSigmaClipPtc)**2
                varDiff *= varFactor

                expIdMask = True
                if np.isnan(muDiff) or np.isnan(varDiff) or (covAstier is None):
                    msg = (f"NaN mean or var, or None cov in amp {ampName} in exposure pair {expId1},"
                           f" {expId2} of detector {detNum}.")
                    self.log.warn(msg)
                    nAmpsNan += 1
                    expIdMask = False
                    covArray = np.full((1, self.config.maximumRangeCovariancesAstier,
                                        self.config.maximumRangeCovariancesAstier), np.nan)
                    covSqrtWeights = np.full_like(covArray, np.nan)

                if (muDiff <= minMeanSignalDict[ampName]) or (muDiff >= maxMeanSignalDict[ampName]):
                    expIdMask = False

                if covAstier is not None:
                    tupleRows = [(muDiff, varDiff) + covRow + (ampNumber, expTime,
                                                               ampName) for covRow in covAstier]
                    tempStructArray = np.array(tupleRows, dtype=tags)
                    covArray, vcov, _ = makeCovArray(tempStructArray,
                                                     self.config.maximumRangeCovariancesAstier)
                    covSqrtWeights = np.nan_to_num(1./np.sqrt(vcov))

                # Correct covArray for sigma clipping:
                # 1) Apply varFactor twice for the whole covariance matrix
                covArray *= varFactor**2
                # 2) But, only once for the variance element of the matrix, covArray[0,0]
                covArray[0, 0] /= varFactor

                partialPtcDataset.setAmpValues(ampName, rawExpTime=[expTime], rawMean=[muDiff],
                                               rawVar=[varDiff], inputExpIdPair=[(expId1, expId2)],
                                               expIdMask=[expIdMask], covArray=covArray,
                                               covSqrtWeights=covSqrtWeights)
            # Use location of exp1 to save PTC dataset from (exp1, exp2) pair.
            # expId1 and expId2, as returned by getInfo().getVisitInfo().getExposureId(),
            # and the exposure IDs stored in inputDims,
            # may have the zero-padded detector number appended at
            # the end (in gen3). A temporary fix is to consider expId//1000 and/or
            # inputDims//1000 (we try also //100 and //10 for other cameras such as DECam).
            # Below, np.where(expId1 == np.array(inputDims)) (and the other analogous
            # comparisons) returns a tuple with a single-element array, so [0][0]
            # is necessary to extract the required index.

            if (match := np.where(expId1 == np.array(inputDims))[0]).shape[0] != 0:
                datasetIndex = match[0]
            elif (match := np.where(expId1/1000 == np.array(inputDims))[0]).shape[0] != 0:
                datasetIndex = match[0]
            elif (match := np.where(expId1/1000 == np.array(inputDims))[0]//1000).shape[0] != 0:
                datasetIndex = match[0]
            elif (match := np.where(expId1//100 == np.array(inputDims))[0]).shape[0] != 0:
                datasetIndex = match[0]
            elif (match := np.where(expId1//100 == np.array(inputDims))[0]//100).shape[0] != 0:
                datasetIndex = match[0]
            elif (match := np.where(expId1//10 == np.array(inputDims))[0]).shape[0] != 0:
                datasetIndex = match[0]
            elif (match := np.where(expId1//10 == np.array(inputDims))[0]//10).shape[0] != 0:
                datasetIndex = match[0]
            else:
                raise RuntimeError("Cannot find appropriate datasetIndex!")

            partialPtcDatasetList[datasetIndex] = partialPtcDataset
            if nAmpsNan == len(ampNames):
                msg = f"NaN mean in all amps of exposure pair {expId1}, {expId2} of detector {detNum}."
                self.log.warn(msg)
        return pipeBase.Struct(
            outputCovariances=partialPtcDatasetList,
        )

    def measureMeanVarCov(self, exposure1, exposure2, region=None, covAstierRealSpace=False):
        """Calculate the mean of each of two exposures and the variance
        and covariance of their difference. The variance is calculated
        via afwMath, and the covariance via the methods in Astier+19
        (appendix A). In theory, var = covariance[0,0]. This should
        be validated, and in the future, we may decide to just keep
        one (covariance).

        Parameters
        ----------
        exposure1 : `lsst.afw.image.exposure.exposure.ExposureF`
            First exposure of flat field pair.
        exposure2 : `lsst.afw.image.exposure.exposure.ExposureF`
            Second exposure of flat field pair.
        region : `lsst.geom.Box2I`, optional
            Region of each exposure where to perform the calculations (e.g, an amplifier).
        covAstierRealSpace : `bool`, optional
            Should the covariannces in Astier+19 be calculated in real space or via FFT?
            See Appendix A of Astier+19.

        Returns
        -------
        mu : `float` or `NaN`
            0.5*(mu1 + mu2), where mu1, and mu2 are the clipped means of the regions in
            both exposures. If either mu1 or m2 are NaN's, the returned value is NaN.
        varDiff : `float` or `NaN`
            Half of the clipped variance of the difference of the regions inthe two input
            exposures. If either mu1 or m2 are NaN's, the returned value is NaN.
        covDiffAstier : `list` or `NaN`
            List with tuples of the form (dx, dy, var, cov, npix), where:
                dx : `int`
                    Lag in x
                dy : `int`
                    Lag in y
                var : `float`
                    Variance at (dx, dy).
                cov : `float`
                    Covariance at (dx, dy).
                nPix : `int`
                    Number of pixel pairs used to evaluate var and cov.
            If either mu1 or m2 are NaN's, the returned value is NaN.
        """

        if region is not None:
            im1Area = exposure1.maskedImage[region]
            im2Area = exposure2.maskedImage[region]
        else:
            im1Area = exposure1.maskedImage
            im2Area = exposure2.maskedImage

        if self.config.binSize > 1:
            im1Area = afwMath.binImage(im1Area, self.config.binSize)
            im2Area = afwMath.binImage(im2Area, self.config.binSize)

        im1MaskVal = exposure1.getMask().getPlaneBitMask(self.config.maskNameList)
        im1StatsCtrl = afwMath.StatisticsControl(self.config.nSigmaClipPtc,
                                                 self.config.nIterSigmaClipPtc,
                                                 im1MaskVal)
        im1StatsCtrl.setNanSafe(True)
        im1StatsCtrl.setAndMask(im1MaskVal)

        im2MaskVal = exposure2.getMask().getPlaneBitMask(self.config.maskNameList)
        im2StatsCtrl = afwMath.StatisticsControl(self.config.nSigmaClipPtc,
                                                 self.config.nIterSigmaClipPtc,
                                                 im2MaskVal)
        im2StatsCtrl.setNanSafe(True)
        im2StatsCtrl.setAndMask(im2MaskVal)

        #  Clipped mean of images; then average of mean.
        mu1 = afwMath.makeStatistics(im1Area, afwMath.MEANCLIP, im1StatsCtrl).getValue()
        mu2 = afwMath.makeStatistics(im2Area, afwMath.MEANCLIP, im2StatsCtrl).getValue()
        if np.isnan(mu1) or np.isnan(mu2):
            self.log.warn(f"Mean of amp in image 1 or 2 is NaN: {mu1}, {mu2}.")
            return np.nan, np.nan, None
        mu = 0.5*(mu1 + mu2)

        # Take difference of pairs
        # symmetric formula: diff = (mu2*im1-mu1*im2)/(0.5*(mu1+mu2))
        temp = im2Area.clone()
        temp *= mu1
        diffIm = im1Area.clone()
        diffIm *= mu2
        diffIm -= temp
        diffIm /= mu

        diffImMaskVal = diffIm.getMask().getPlaneBitMask(self.config.maskNameList)
        diffImStatsCtrl = afwMath.StatisticsControl(self.config.nSigmaClipPtc,
                                                    self.config.nIterSigmaClipPtc,
                                                    diffImMaskVal)
        diffImStatsCtrl.setNanSafe(True)
        diffImStatsCtrl.setAndMask(diffImMaskVal)

        # Variance calculation via afwMath
        varDiff = 0.5*(afwMath.makeStatistics(diffIm, afwMath.VARIANCECLIP, diffImStatsCtrl).getValue())

        # Covariances calculations
        # Get the pixels that were not clipped
        varClip = afwMath.makeStatistics(diffIm, afwMath.VARIANCECLIP, diffImStatsCtrl).getValue()
        meanClip = afwMath.makeStatistics(diffIm, afwMath.MEANCLIP, diffImStatsCtrl).getValue()
        cut = meanClip + self.config.nSigmaClipPtc*np.sqrt(varClip)
        unmasked = np.where(np.fabs(diffIm.image.array) <= cut, 1, 0)

        # Get the pixels in the mask planes of teh differenc eimage that were ignored
        # by the clipping algorithm
        wDiff = np.where(diffIm.getMask().getArray() == 0, 1, 0)
        # Combine the two sets of pixels ('1': use; '0': don't use) into a final weight matrix
        # to be used in the covariance calculations below.
        w = unmasked*wDiff

        if np.sum(w) < self.config.minNumberGoodPixelsForCovariance:
            self.log.warn(f"Number of good points for covariance calculation ({np.sum(w)}) is less "
                          f"(than threshold {self.config.minNumberGoodPixelsForCovariance})")
            return np.nan, np.nan, None

        maxRangeCov = self.config.maximumRangeCovariancesAstier
        if covAstierRealSpace:
            # Calculate  covariances in real space.
            covDiffAstier = computeCovDirect(diffIm.image.array, w, maxRangeCov)
        else:
            # Calculate covariances via FFT (default).
            shapeDiff = np.array(diffIm.image.array.shape)
            # Calculate the sizes of FFT dimensions.
            s = shapeDiff + maxRangeCov
            tempSize = np.array(np.log(s)/np.log(2.)).astype(int)
            fftSize = np.array(2**(tempSize+1)).astype(int)
            fftShape = (fftSize[0], fftSize[1])

            c = CovFastFourierTransform(diffIm.image.array, w, fftShape, maxRangeCov)
            covDiffAstier = c.reportCovFastFourierTransform(maxRangeCov)

        # Compare Cov[0,0] and afwMath.VARIANCECLIP
        # covDiffAstier[0] is the Cov[0,0] element, [3] is the variance, and there's a factor of 0.5
        # difference with afwMath.VARIANCECLIP.
        thresholdPercentage = self.config.thresholdDiffAfwVarVsCov00
        fractionalDiff = 100*np.fabs(1 - varDiff/(covDiffAstier[0][3]*0.5))
        if fractionalDiff >= thresholdPercentage:
            self.log.warn("Absolute fractional difference between afwMatch.VARIANCECLIP and Cov[0,0] "
                          f"is more than {thresholdPercentage}%: {fractionalDiff}")

        return mu, varDiff, covDiffAstier
