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
import time

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage

from lsst.geom import Point2D
from lsst.log import Log
from astro_metadata_translator import merge_headers, ObservationGroup
from astro_metadata_translator.serialize import dates_to_fits


# CalibStatsConfig/CalibStatsTask from pipe_base/constructCalibs.py
class CalibStatsConfig(pexConfig.Config):
    """Parameters controlling the measurement of background statistics.
    """
    stat = pexConfig.Field(
        dtype=str,
        default='MEANCLIP',
        doc="Statistic name to use to estimate background (from lsst.afw.math)",
    )
    clip = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Clipping threshold for background",
    )
    nIter = pexConfig.Field(
        dtype=int,
        default=3,
        doc="Clipping iterations for background",
    )
    mask = pexConfig.ListField(
        dtype=str,
        default=["DETECTED", "BAD", "NO_DATA"],
        doc="Mask planes to reject",
    )


class CalibStatsTask(pipeBase.Task):
    """Measure statistics on the background

    This can be useful for scaling the background, e.g., for flats and fringe frames.
    """
    ConfigClass = CalibStatsConfig

    def run(self, exposureOrImage):
        """Measure a particular statistic on an image (of some sort).

        Parameters
        ----------
        exposureOrImage : `lsst.afw.image.Exposure`, `lsst.afw.image.MaskedImage`, or `lsst.afw.image.Image`
           Exposure or image to calculate statistics on.

        Returns
        -------
        results : float
           Resulting statistic value.
        """
        stats = afwMath.StatisticsControl(self.config.clip, self.config.nIter,
                                          afwImage.Mask.getPlaneBitMask(self.config.mask))
        try:
            image = exposureOrImage.getMaskedImage()
        except Exception:
            try:
                image = exposureOrImage.getImage()
            except Exception:
                image = exposureOrImage
        statType = afwMath.stringToStatisticsProperty(self.config.stat)
        return afwMath.makeStatistics(image, statType, stats).getValue()


class CalibCombineConnections(pipeBase.PipelineTaskConnections,
                              dimensions=("instrument", "detector")):
    inputExps = cT.Input(
        name="cpInputs",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "exposure"),
        multiple=True,
    )
    inputScales = cT.Input(
        name="cpScales",
        doc="Input scale factors to use.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", ),
        multiple=False,
    )

    outputData = cT.Output(
        name="cpProposal",
        doc="Output combined proposed calibration to be validated and certified..",
        storageClass="ExposureF",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config and config.exposureScaling != 'InputList':
            self.inputs.discard("inputScales")


# CalibCombineConfig/CalibCombineTask from pipe_base/constructCalibs.py
class CalibCombineConfig(pipeBase.PipelineTaskConfig,
                         pipelineConnections=CalibCombineConnections):
    """Configuration for combining calib exposures.
    """
    calibrationType = pexConfig.Field(
        dtype=str,
        default="calibration",
        doc="Name of calibration to be generated.",
    )

    exposureScaling = pexConfig.ChoiceField(
        dtype=str,
        allowed={
            "Unity": "Do not scale inputs.  Scale factor is 1.0.",
            "ExposureTime": "Scale inputs by their exposure time.",
            "DarkTime": "Scale inputs by their dark time.",
            "MeanStats": "Scale inputs based on their mean values.",
            "InputList": "Scale inputs based on a list of values.",
        },
        default="Unity",
        doc="Scaling to be applied to each input exposure.",
    )
    scalingLevel = pexConfig.ChoiceField(
        dtype=str,
        allowed={
            "DETECTOR": "Scale by detector.",
            "AMP": "Scale by amplifier.",
        },
        default="DETECTOR",
        doc="Region to scale.",
    )
    maxVisitsToCalcErrorFromInputVariance = pexConfig.Field(
        dtype=int,
        default=5,
        doc="Maximum number of visits to estimate variance from input variance, not per-pixel spread",
    )

    doVignette = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Copy vignette polygon to output and censor vignetted pixels?"
    )

    mask = pexConfig.ListField(
        dtype=str,
        default=["SAT", "DETECTED", "INTRP"],
        doc="Mask planes to respect",
    )
    combine = pexConfig.Field(
        dtype=str,
        default='MEANCLIP',
        doc="Statistic name to use for combination (from lsst.afw.math)",
    )
    clip = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Clipping threshold for combination",
    )
    nIter = pexConfig.Field(
        dtype=int,
        default=3,
        doc="Clipping iterations for combination",
    )
    stats = pexConfig.ConfigurableField(
        target=CalibStatsTask,
        doc="Background statistics configuration",
    )


class CalibCombineTask(pipeBase.PipelineTask,
                       pipeBase.CmdLineTask):
    """Task to combine calib exposures."""
    ConfigClass = CalibCombineConfig
    _DefaultName = 'cpCombine'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("stats")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        dimensions = [exp.dataId.byName() for exp in inputRefs.inputExps]
        inputs['inputDims'] = dimensions

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputExps, inputScales=None, inputDims=None):
        """Combine calib exposures for a single detector.

        Parameters
        ----------
        inputExps : `list` [`lsst.afw.image.Exposure`]
            Input list of exposures to combine.
        inputScales : `dict` [`dict` [`dict` [`float`]]], optional
            Dictionary of scales, indexed by detector (`int`),
            amplifier (`int`), and exposure (`int`).  Used for
            'inputList' scaling.
        inputDims : `list` [`dict`]
            List of dictionaries of input data dimensions/values.
            Each list entry should contain:

            ``"exposure"``
                exposure id value (`int`)
            ``"detector"``
                detector id value (`int`)

        Returns
        -------
        combinedExp : `lsst.afw.image.Exposure`
            Final combined exposure generated from the inputs.

        Raises
        ------
        RuntimeError
            Raised if no input data is found.  Also raised if
            config.exposureScaling == InputList, and a necessary scale
            was not found.
        """
        width, height = self.getDimensions(inputExps)
        stats = afwMath.StatisticsControl(self.config.clip, self.config.nIter,
                                          afwImage.Mask.getPlaneBitMask(self.config.mask))
        numExps = len(inputExps)
        if numExps < 1:
            raise RuntimeError("No valid input data")
        if numExps < self.config.maxVisitsToCalcErrorFromInputVariance:
            stats.setCalcErrorFromInputVariance(True)

        detectorList = [exp.getDetector() for exp in inputExps]
        if None in detectorList:
            self.log.warn("Not all input detectors defined.")
        numDetectors = len(set([det.getId() for det in detectorList if det is not None]))
        if numDetectors > 1:
            raise RuntimeError("Input data contains multiple detectors.")

        # Create output exposure for combined data.
        combined = afwImage.MaskedImageF(width, height)
        combinedExp = afwImage.makeExposure(combined)

        # Apply scaling:
        expScales = []
        if inputDims is None:
            inputDims = [dict() for i in inputExps]

        for index, (exp, dims) in enumerate(zip(inputExps, inputDims)):
            scale = 1.0
            if exp is None:
                self.log.warn("Input %d is None (%s); unable to scale exp.", index, dims)
                continue

            if self.config.exposureScaling == "ExposureTime":
                scale = exp.getInfo().getVisitInfo().getExposureTime()
            elif self.config.exposureScaling == "DarkTime":
                scale = exp.getInfo().getVisitInfo().getDarkTime()
            elif self.config.exposureScaling == "MeanStats":
                scale = self.stats.run(exp)
            elif self.config.exposureScaling == "InputList":
                visitId = dims.get('exposure', None)
                detectorId = dims.get('detector', None)
                if visitId is None or detectorId is None:
                    raise RuntimeError(f"Could not identify scaling for input {index} ({dims})")
                if detectorId not in inputScales['expScale']:
                    raise RuntimeError(f"Could not identify a scaling for input {index}"
                                       f" detector {detectorId}")

                if self.config.scalingLevel == "DETECTOR":
                    if visitId not in inputScales['expScale'][detectorId]:
                        raise RuntimeError(f"Could not identify a scaling for input {index}"
                                           f"detector {detectorId} visit {visitId}")
                    scale = inputScales['expScale'][detectorId][visitId]
                elif self.config.scalingLevel == 'AMP':
                    scale = [inputScales['expScale'][detectorId][amp.getName()][visitId]
                             for amp in exp.getDetector()]
                else:
                    raise RuntimeError(f"Unknown scaling level: {self.config.scalingLevel}")
            elif self.config.exposureScaling == 'Unity':
                scale = 1.0
            else:
                raise RuntimeError(f"Unknown scaling type: {self.config.exposureScaling}.")

            expScales.append(scale)
            self.log.info("Scaling input %d by %s", index, scale)
            self.applyScale(exp, scale)

        self.combine(combined, inputExps, stats)

        self.interpolateNans(combined)

        if self.config.doVignette:
            polygon = inputExps[0].getInfo().getValidPolygon()
            VignetteExposure(combined, polygon=polygon, doUpdateMask=True,
                             doSetValue=True, vignetteValue=0.0)

        # Combine headers
        self.combineHeaders(inputExps, combinedExp,
                            calibType=self.config.calibrationType, scales=expScales)

        # Set the detector
        inputDetector = inputExps[0].getDetector()
        combinedExp.setDetector(inputDetector)

        # Return
        return pipeBase.Struct(
            outputData=combinedExp,
        )

    def getDimensions(self, expList):
        """Get dimensions of the inputs.

        Parameters
        ----------
        expList : `list` [`lsst.afw.image.Exposure`]
            Exps to check the sizes of.

        Returns
        -------
        width, height : `int`
            Unique set of input dimensions.
        """
        dimList = [exp.getDimensions() for exp in expList if exp is not None]
        return self.getSize(dimList)

    def getSize(self, dimList):
        """Determine a consistent size, given a list of image sizes.

        Parameters
        -----------
        dimList : iterable of `tuple` (`int`, `int`)
            List of dimensions.

        Raises
        ------
        RuntimeError
            If input dimensions are inconsistent.

        Returns
        --------
        width, height : `int`
            Common dimensions.
        """
        dim = set((w, h) for w, h in dimList)
        if len(dim) != 1:
            raise RuntimeError("Inconsistent dimensions: %s" % dim)
        return dim.pop()

    def applyScale(self, exposure, scale=None):
        """Apply scale to input exposure.

        This implementation applies a flux scaling: the input exposure is
        divided by the provided scale.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to scale.
        scale : `float` or `list` [`float`], optional
            Constant scale to divide the exposure by.
        """
        if scale is not None:
            mi = exposure.getMaskedImage()
            if isinstance(scale, list):
                for amp, ampScale in zip(exposure.getDetector(), scale):
                    ampIm = mi[amp.getBBox()]
                    ampIm /= ampScale
            else:
                mi /= scale

    def combine(self, target, expList, stats):
        """Combine multiple images.

        Parameters
        ----------
        target : `lsst.afw.image.Exposure`
            Output exposure to construct.
        expList : `list` [`lsst.afw.image.Exposure`]
            Input exposures to combine.
        stats : `lsst.afw.math.StatisticsControl`
            Control explaining how to combine the input images.
        """
        images = [img.getMaskedImage() for img in expList if img is not None]
        combineType = afwMath.stringToStatisticsProperty(self.config.combine)
        afwMath.statisticsStack(target, images, combineType, stats)

    def combineHeaders(self, expList, calib, calibType="CALIB", scales=None):
        """Combine input headers to determine the set of common headers,
        supplemented by calibration inputs.

        Parameters
        ----------
        expList : `list` of `lsst.afw.image.Exposure`
            Input list of exposures to combine.
        calib : `lsst.afw.image.Exposure`
            Output calibration to construct headers for.
        calibType: `str`, optional
            OBSTYPE the output should claim.
        scales: `list` of `float`, optional
            Scale values applied to each input to record.

        Returns
        -------
        header : `lsst.daf.base.PropertyList`
            Constructed header.
        """
        # Header
        header = calib.getMetadata()
        header.set("OBSTYPE", calibType)

        # Keywords we care about
        comments = {"TIMESYS": "Time scale for all dates",
                    "DATE-OBS": "Start date of earliest input observation",
                    "MJD-OBS": "[d] Start MJD of earliest input observation",
                    "DATE-END": "End date of oldest input observation",
                    "MJD-END": "[d] End MJD of oldest input observation",
                    "MJD-AVG": "[d] MJD midpoint of all input observations",
                    "DATE-AVG": "Midpoint date of all input observations"}

        # Creation date
        now = time.localtime()
        calibDate = time.strftime("%Y-%m-%d", now)
        calibTime = time.strftime("%X %Z", now)
        header.set("CALIB_CREATE_DATE", calibDate)
        header.set("CALIB_CREATE_TIME", calibTime)

        # Merge input headers
        inputHeaders = [exp.getMetadata() for exp in expList if exp is not None]
        merged = merge_headers(inputHeaders, mode='drop')
        for k, v in merged.items():
            if k not in header:
                md = expList[0].getMetadata()
                comment = md.getComment(k) if k in md else None
                header.set(k, v, comment=comment)

        # Construct list of visits
        visitInfoList = [exp.getInfo().getVisitInfo() for exp in expList if exp is not None]
        for i, visit in enumerate(visitInfoList):
            if visit is None:
                continue
            header.set("CPP_INPUT_%d" % (i,), visit.getExposureId())
            header.set("CPP_INPUT_DATE_%d" % (i,), str(visit.getDate()))
            header.set("CPP_INPUT_EXPT_%d" % (i,), visit.getExposureTime())
            if scales is not None:
                header.set("CPP_INPUT_SCALE_%d" % (i,), scales[i])

        # Not yet working: DM-22302
        # Create an observation group so we can add some standard headers
        # independent of the form in the input files.
        # Use try block in case we are dealing with unexpected data headers
        try:
            group = ObservationGroup(visitInfoList, pedantic=False)
        except Exception:
            self.log.warn("Exception making an obs group for headers. Continuing.")
            # Fall back to setting a DATE-OBS from the calibDate
            dateCards = {"DATE-OBS": "{}T00:00:00.00".format(calibDate)}
            comments["DATE-OBS"] = "Date of start of day of calibration midpoint"
        else:
            oldest, newest = group.extremes()
            dateCards = dates_to_fits(oldest.datetime_begin, newest.datetime_end)

        for k, v in dateCards.items():
            header.set(k, v, comment=comments.get(k, None))

        return header

    def interpolateNans(self, exp):
        """Interpolate over NANs in the combined image.

        NANs can result from masked areas on the CCD.  We don't want them getting
        into our science images, so we replace them with the median of the image.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            Exp to check for NaNs.
        """
        array = exp.getImage().getArray()
        bad = np.isnan(array)

        median = np.median(array[np.logical_not(bad)])
        count = np.sum(np.logical_not(bad))
        array[bad] = median
        if count > 0:
            self.log.warn("Found %s NAN pixels", count)


# Create versions of the Connections, Config, and Task that support filter constraints.
class CalibCombineByFilterConnections(CalibCombineConnections,
                                      dimensions=("instrument", "detector", "physical_filter")):
    inputScales = cT.Input(
        name="cpFilterScales",
        doc="Input scale factors to use.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "physical_filter"),
        multiple=False,
    )

    outputData = cT.Output(
        name="cpFilterProposal",
        doc="Output combined proposed calibration to be validated and certified.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector", "physical_filter"),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config and config.exposureScaling != 'InputList':
            self.inputs.discard("inputScales")


class CalibCombineByFilterConfig(CalibCombineConfig,
                                 pipelineConnections=CalibCombineByFilterConnections):
    pass


class CalibCombineByFilterTask(CalibCombineTask):
    """Task to combine calib exposures."""
    ConfigClass = CalibCombineByFilterConfig
    _DefaultName = 'cpFilterCombine'
    pass


def VignetteExposure(exposure, polygon=None,
                     doUpdateMask=True, maskPlane='BAD',
                     doSetValue=False, vignetteValue=0.0,
                     log=None):
    """Apply vignetted polygon to image pixels.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        Image to be updated.
    doUpdateMask : `bool`, optional
        Update the exposure mask for vignetted area?
    maskPlane : `str`, optional,
        Mask plane to assign.
    doSetValue : `bool`, optional
        Set image value for vignetted area?
    vignetteValue : `float`, optional
        Value to assign.
    log : `lsst.log.Log`, optional
        Log to write to.

    Raises
    ------
    RuntimeError
        Raised if no valid polygon exists.
    """
    polygon = polygon if polygon else exposure.getInfo().getValidPolygon()
    if not polygon:
        raise RuntimeError("Could not find valid polygon!")
    log = log if log else Log.getLogger(__name__.partition(".")[2])

    fullyIlluminated = True
    for corner in exposure.getBBox().getCorners():
        if not polygon.contains(Point2D(corner)):
            fullyIlluminated = False

    log.info("Exposure is fully illuminated? %s", fullyIlluminated)

    if not fullyIlluminated:
        # Scan pixels.
        mask = exposure.getMask()
        numPixels = mask.getBBox().getArea()

        xx, yy = np.meshgrid(np.arange(0, mask.getWidth(), dtype=int),
                             np.arange(0, mask.getHeight(), dtype=int))

        vignMask = np.array([not polygon.contains(Point2D(x, y)) for x, y in
                             zip(xx.reshape(numPixels), yy.reshape(numPixels))])
        vignMask = vignMask.reshape(mask.getHeight(), mask.getWidth())

        if doUpdateMask:
            bitMask = mask.getPlaneBitMask(maskPlane)
            maskArray = mask.getArray()
            maskArray[vignMask] |= bitMask
        if doSetValue:
            imageArray = exposure.getImage().getArray()
            imageArray[vignMask] = vignetteValue
        log.info("Exposure contains %d vignetted pixels.",
                 np.count_nonzero(vignMask))
