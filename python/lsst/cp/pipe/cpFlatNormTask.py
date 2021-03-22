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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
from collections import defaultdict

import lsst.afw.math as afwMath
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.cp.pipe.cpCombine import VignetteExposure
from lsst.cp.pipe.utils import ddict2dict

from ._lookupStaticCalibration import lookupStaticCalibration

__all__ = ["CpFlatMeasureTask", "CpFlatMeasureTaskConfig",
           "CpFlatNormalizationTask", "CpFlatNormalizationTaskConfig"]


class CpFlatMeasureConnections(pipeBase.PipelineTaskConnections,
                               dimensions=("instrument", "exposure", "detector")):
    inputExp = cT.Input(
        name="postISRCCD",
        doc="Input exposure to measure statistics from.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    outputStats = cT.Output(
        name="flatStats",
        doc="Output statistics to write.",
        storageClass="PropertyList",
        dimensions=("instrument", "exposure", "detector"),
    )


class CpFlatMeasureTaskConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=CpFlatMeasureConnections):
    maskNameList = pexConfig.ListField(
        dtype=str,
        doc="Mask list to exclude from statistics calculations.",
        default=['DETECTED', 'BAD', 'NO_DATA'],
    )
    doVignette = pexConfig.Field(
        dtype=bool,
        doc="Mask vignetted regions?",
        default=True,
    )
    numSigmaClip = pexConfig.Field(
        dtype=float,
        doc="Rejection threshold (sigma) for statistics clipping.",
        default=3.0,
    )
    clipMaxIter = pexConfig.Field(
        dtype=int,
        doc="Max number of clipping iterations to apply.",
        default=3,
    )


class CpFlatMeasureTask(pipeBase.PipelineTask,
                        pipeBase.CmdLineTask):
    """Apply extra masking and measure image statistics.
    """
    ConfigClass = CpFlatMeasureTaskConfig
    _DefaultName = "cpFlatMeasure"

    def run(self, inputExp):
        """Mask ISR processed FLAT exposures to ensure consistent statistics.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Post-ISR processed exposure to measure.

        Returns
        -------
        outputStats : `lsst.daf.base.PropertyList`
            List containing the statistics.
        """
        if self.config.doVignette:
            VignetteExposure(inputExp, doUpdateMask=True, doSetValue=False, log=self.log)
        mask = inputExp.getMask()
        maskVal = mask.getPlaneBitMask(self.config.maskNameList)
        statsControl = afwMath.StatisticsControl(self.config.numSigmaClip,
                                                 self.config.clipMaxIter,
                                                 maskVal)
        statsControl.setAndMask(maskVal)

        outputStats = dafBase.PropertyList()

        # Detector level:
        stats = afwMath.makeStatistics(inputExp.getMaskedImage(),
                                       afwMath.MEANCLIP | afwMath.STDEVCLIP | afwMath.NPOINT,
                                       statsControl)
        outputStats['DETECTOR_MEDIAN'] = stats.getValue(afwMath.MEANCLIP)
        outputStats['DETECTOR_SIGMA'] = stats.getValue(afwMath.STDEVCLIP)
        outputStats['DETECTOR_N'] = stats.getValue(afwMath.NPOINT)
        self.log.info("Stats: median=%f sigma=%f n=%d",
                      outputStats['DETECTOR_MEDIAN'],
                      outputStats['DETECTOR_SIGMA'],
                      outputStats['DETECTOR_N'])

        # AMP LEVEL:
        for ampIdx, amp in enumerate(inputExp.getDetector()):
            ampName = amp.getName()
            ampExp = inputExp.Factory(inputExp, amp.getBBox())
            stats = afwMath.makeStatistics(ampExp.getMaskedImage(),
                                           afwMath.MEANCLIP | afwMath.STDEVCLIP | afwMath.NPOINT,
                                           statsControl)
            outputStats[f'AMP_NAME_{ampIdx}'] = ampName
            outputStats[f'AMP_MEDIAN_{ampIdx}'] = stats.getValue(afwMath.MEANCLIP)
            outputStats[f'AMP_SIGMA_{ampIdx}'] = stats.getValue(afwMath.STDEVCLIP)
            outputStats[f'AMP_N_{ampIdx}'] = stats.getValue(afwMath.NPOINT)

        return pipeBase.Struct(
            outputStats=outputStats
        )


class CpFlatNormalizationConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("instrument", "physical_filter")):
    inputMDs = cT.Input(
        name="cpFlatProc_metadata",
        doc="Input metadata for each visit/detector in input set.",
        storageClass="PropertyList",
        dimensions=("instrument", "physical_filter", "detector", "exposure"),
        multiple=True,
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Input camera to use for gain lookup.",
        storageClass="Camera",
        dimensions=("instrument",),
        lookupFunction=lookupStaticCalibration,
        isCalibration=True,
    )

    outputScales = cT.Output(
        name="cpFlatNormScales",
        doc="Output combined proposed calibration.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "physical_filter"),
    )


class CpFlatNormalizationTaskConfig(pipeBase.PipelineTaskConfig,
                                    pipelineConnections=CpFlatNormalizationConnections):
    level = pexConfig.ChoiceField(
        dtype=str,
        doc="Which level to apply normalizations.",
        default='DETECTOR',
        allowed={
            'DETECTOR': "Correct using full detector statistics.",
            'AMP': "Correct using individual amplifiers.",
        },
    )
    scaleMaxIter = pexConfig.Field(
        dtype=int,
        doc="Max number of iterations to use in scale solver.",
        default=10,
    )


class CpFlatNormalizationTask(pipeBase.PipelineTask,
                              pipeBase.CmdLineTask):
    """Rescale merged flat frames to remove unequal screen illumination.
    """
    ConfigClass = CpFlatNormalizationTaskConfig
    _DefaultName = "cpFlatNorm"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        # Use the dimensions of the inputs for generating
        # output scales.
        dimensions = [exp.dataId.byName() for exp in inputRefs.inputMDs]
        inputs['inputDims'] = dimensions

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputMDs, inputDims, camera):
        """Normalize FLAT exposures to a consistent level.

        Parameters
        ----------
        inputMDs : `list` [`lsst.daf.base.PropertyList`]
            Amplifier-level metadata used to construct scales.
        inputDims : `list` [`dict`]
            List of dictionaries of input data dimensions/values.
            Each list entry should contain:

            ``"exposure"``
                exposure id value (`int`)
            ``"detector"``
                detector id value (`int`)

        Returns
        -------
        outputScales : `dict` [`dict` [`dict` [`float`]]]
            Dictionary of scales, indexed by detector (`int`),
            amplifier (`int`), and exposure (`int`).

        Raises
        ------
        KeyError
            Raised if the input dimensions do not contain detector and
            exposure, or if the metadata does not contain the expected
            statistic entry.
        """
        expSet = sorted(set([d['exposure'] for d in inputDims]))
        detSet = sorted(set([d['detector'] for d in inputDims]))

        expMap = {exposureId: idx for idx, exposureId in enumerate(expSet)}
        detMap = {detectorId: idx for idx, detectorId in enumerate(detSet)}

        nExp = len(expSet)
        nDet = len(detSet)
        if self.config.level == 'DETECTOR':
            bgMatrix = np.zeros((nDet, nExp))
            bgCounts = np.ones((nDet, nExp))
        elif self.config.level == 'AMP':
            nAmp = len(camera[detSet[0]])
            bgMatrix = np.zeros((nDet * nAmp, nExp))
            bgCounts = np.ones((nDet * nAmp, nExp))

        for inMetadata, inDimensions in zip(inputMDs, inputDims):
            try:
                exposureId = inDimensions['exposure']
                detectorId = inDimensions['detector']
            except Exception as e:
                raise KeyError("Cannot find expected dimensions in %s" % (inDimensions, )) from e

            if self.config.level == 'DETECTOR':
                detIdx = detMap[detectorId]
                expIdx = expMap[exposureId]
                try:
                    value = inMetadata.get('DETECTOR_MEDIAN')
                    count = inMetadata.get('DETECTOR_N')
                except Exception as e:
                    raise KeyError("Cannot read expected metadata string.") from e

                if np.isfinite(value):
                    bgMatrix[detIdx][expIdx] = value
                    bgCounts[detIdx][expIdx] = count
                else:
                    bgMatrix[detIdx][expIdx] = np.nan
                    bgCounts[detIdx][expIdx] = 1

            elif self.config.level == 'AMP':
                detector = camera[detectorId]
                nAmp = len(detector)

                detIdx = detMap[detectorId] * nAmp
                expIdx = expMap[exposureId]

                for ampIdx, amp in enumerate(detector):
                    try:
                        value = inMetadata.get(f'AMP_MEDIAN_{ampIdx}')
                        count = inMetadata.get(f'AMP_N_{ampIdx}')
                    except Exception as e:
                        raise KeyError("cannot read expected metadata string.") from e

                    detAmpIdx = detIdx + ampIdx
                    if np.isfinite(value):
                        bgMatrix[detAmpIdx][expIdx] = value
                        bgCounts[detAmpIdx][expIdx] = count
                    else:
                        bgMatrix[detAmpIdx][expIdx] = np.nan
                        bgMatrix[detAmpIdx][expIdx] = 1

        scaleResult = self.measureScales(bgMatrix, bgCounts, iterations=self.config.scaleMaxIter)
        expScales = scaleResult.expScales
        detScales = scaleResult.detScales

        outputScales = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

        # Note that the enumerated "detId"/"expId" here index the
        # "detScales" and "expScales" arrays.
        if self.config.level == 'DETECTOR':
            for detIdx, det in enumerate(detSet):
                for amp in camera[det]:
                    for expIdx, exp in enumerate(expSet):
                        outputScales['expScale'][det][amp.getName()][exp] = expScales[expIdx].tolist()
                outputScales['detScale'][det] = detScales[detIdx].tolist()
        elif self.config.level == 'AMP':
            for detIdx, det in enumerate(detSet):
                for ampIdx, amp in enumerate(camera[det]):
                    for expIdx, exp in enumerate(expSet):
                        outputScales['expScale'][det][amp.getName()][exp] = expScales[expIdx].tolist()
                    detAmpIdx = detIdx + ampIdx
                    outputScales['detScale'][det][amp.getName()] = detScales[detAmpIdx].tolist()

        return pipeBase.Struct(
            outputScales=ddict2dict(outputScales),
        )

    def measureScales(self, bgMatrix, bgCounts=None, iterations=10):
        """Convert backgrounds to exposure and detector components.

        Parameters
        ----------
        bgMatrix : `np.ndarray`, (nDetectors, nExposures)
            Input backgrounds indexed by exposure (axis=0) and
            detector (axis=1).
        bgCounts : `np.ndarray`, (nDetectors, nExposures), optional
            Input pixel counts used to in measuring bgMatrix, indexed
            identically.
        iterations : `int`, optional
            Number of iterations to use in decomposition.

        Returns
        -------
        scaleResult : `lsst.pipe.base.Struct`
            Result struct containing fields:

            ``vectorE``
                Output E vector of exposure level scalings
                (`np.array`, (nExposures)).
            ``vectorG``
                Output G vector of detector level scalings
                (`np.array`, (nExposures)).
            ``bgModel``
                Expected model bgMatrix values, calculated from E and G
                (`np.ndarray`, (nDetectors, nExposures)).

        Notes
        -----

        The set of background measurements B[exposure, detector] of
        flat frame data should be defined by a "Cartesian" product of
        two vectors, E[exposure] and G[detector].  The E vector
        represents the total flux incident on the focal plane.  In a
        perfect camera, this is simply the sum along the columns of B
        (np.sum(B, axis=0)).

        However, this simple model ignores differences in detector
        gains, the vignetting of the detectors, and the illumination
        pattern of the source lamp.  The G vector describes these
        detector dependent differences, which should be identical over
        different exposures.  For a perfect lamp of unit total
        intensity, this is simply the sum along the rows of B
        (np.sum(B, axis=1)).  This algorithm divides G by the total
        flux level, to provide the relative (not absolute) scales
        between detectors.

        The algorithm here, from pipe_drivers/constructCalibs.py and
        from there from Eugene Magnier/PanSTARRS [1]_, attempts to
        iteratively solve this decomposition from initial "perfect" E
        and G vectors.  The operation is performed in log space to
        reduce the multiply and divides to linear additions and
        subtractions.

        References
        ----------
        .. [1] https://svn.pan-starrs.ifa.hawaii.edu/trac/ipp/browser/trunk/psModules/src/detrend/pmFlatNormalize.c  # noqa: E501

        """
        numExps = bgMatrix.shape[1]
        numChips = bgMatrix.shape[0]
        if bgCounts is None:
            bgCounts = np.ones_like(bgMatrix)

        logMeas = np.log(bgMatrix)
        logMeas = np.ma.masked_array(logMeas, ~np.isfinite(logMeas))
        logG = np.zeros(numChips)
        logE = np.array([np.average(logMeas[:, iexp] - logG,
                                    weights=bgCounts[:, iexp]) for iexp in range(numExps)])

        for iter in range(iterations):
            logG = np.array([np.average(logMeas[ichip, :] - logE,
                                        weights=bgCounts[ichip, :]) for ichip in range(numChips)])

            bad = np.isnan(logG)
            if np.any(bad):
                logG[bad] = logG[~bad].mean()

            logE = np.array([np.average(logMeas[:, iexp] - logG,
                                        weights=bgCounts[:, iexp]) for iexp in range(numExps)])
            fluxLevel = np.average(np.exp(logG), weights=np.sum(bgCounts, axis=1))

            logG -= np.log(fluxLevel)
            self.log.debug(f"ITER {iter}: Flux: {fluxLevel}")
            self.log.debug(f"Exps: {np.exp(logE)}")
            self.log.debug(f"{np.mean(logG)}")

        logE = np.array([np.average(logMeas[:, iexp] - logG,
                                    weights=bgCounts[:, iexp]) for iexp in range(numExps)])

        bgModel = np.exp(logE[np.newaxis, :] - logG[:, np.newaxis])
        return pipeBase.Struct(
            expScales=np.exp(logE),
            detScales=np.exp(logG),
            bgModel=bgModel,
        )
