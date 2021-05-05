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

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage

from lsst.pipe.drivers.background import (FocalPlaneBackground, MaskObjectsTask, SkyMeasurementTask,
                                          FocalPlaneBackgroundConfig, BackgroundConfig)
from lsst.daf.base import PropertyList
from ._lookupStaticCalibration import lookupStaticCalibration

__all__ = ['CpSkyImageTask', 'CpSkyImageConfig',
           'CpSkyScaleMeasureTask', 'CpSkyScaleMeasureConfig',
           'CpSkySubtractBackgroundTask', 'CpSkySubtractBackgroundConfig',
           'CpSkyCombineTask', 'CpSkyCombineConfig']


class CpSkyImageConnections(pipeBase.PipelineTaskConnections,
                            dimensions=("instrument", "physical_filter", "exposure", "detector")):
    inputExp = cT.Input(
        name="cpSkyIsr",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Input camera to use for geometry.",
        storageClass="Camera",
        dimensions=("instrument",),
        lookupFunction=lookupStaticCalibration,
        isCalibration=True,
    )

    maskedExp = cT.Output(
        name="cpSkyMaskedIsr",
        doc="Output masked post-ISR exposure.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    maskedBkg = cT.Output(
        name="cpSkyDetectorBackground",
        doc="Initial background model from one image.",
        storageClass="FocalPlaneBackground",
        dimensions=("instrument", "exposure", "detector"),
    )


class CpSkyImageConfig(pipeBase.PipelineTaskConfig,
                       pipelineConnections=CpSkyImageConnections):
    maskTask = pexConfig.ConfigurableField(
        target=MaskObjectsTask,
        doc="Object masker to use.",
    )

    maskThresh = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="k-sigma threshold for masking pixels.",
    )
    maskList = pexConfig.ListField(
        dtype=str,
        default=["DETECTED", "BAD", "NO_DATA", "SAT"],
        doc="Mask planes to reject.",
    )

    largeScaleBackground = pexConfig.ConfigField(
        dtype=FocalPlaneBackgroundConfig,
        doc="Large-scale background configuration",
    )

    def setDefaults(self):
        # self.largeScaleBackground.xSize = 256
        # self.largeScaleBackground.ySize = 256
        # obs_subaru values
        self.largeScaleBackground.xSize = 122.88
        self.largeScaleBackground.ySize = 122.88
        self.largeScaleBackground.pixelSize = 0.015
        self.largeScaleBackground.minFrac = 0.1
        self.largeScaleBackground.mask = ['BAD', 'SAT', 'INTRP', 'DETECTED', 'DETECTED_NEGATIVE',
                                          'EDGE', 'NO_DATA']


# cpSkyMaskedIsr_{per exposure,detector} = map(applyMask, list(cpSkyProc))
class CpSkyImageTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Mask the detections on the postISRCCD
    """
    ConfigClass = CpSkyImageConfig
    _DefaultName = "CpSkyImage"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("maskTask")

    def run(self, inputExp, camera):
        """Mask the detections on the postISRCCD.

        CZW: Write docstring.
        """
        # Duplicate ST.processSingleBackground
        # Except: check if a detector is fully masked to avoid
        # self.maskTask raising.
        currentMask = inputExp.getMask()
        badMask = currentMask.getPlaneBitMask(self.config.maskList)
        if (currentMask.getArray() & badMask).all():
            self.log.warn("All pixels are masked!")
        else:
            self.maskTask.run(inputExp, self.config.maskList)

        # Duplicate ST.measureBackground
        bgModel = FocalPlaneBackground.fromCamera(self.config.largeScaleBackground, camera)
        bgModel.addCcd(inputExp)

        return pipeBase.Struct(
            maskedExp=inputExp,
            maskedBkg=bgModel,
        )


class CpSkyScaleMeasureConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("instrument", "physical_filter", "exposure")):
    inputBkgs = cT.Input(
        name="cpSkyDetectorBackground",
        doc="Initial background model from one exposure/detector",
        storageClass="FocalPlaneBackground",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True
    )

    outputBkg = cT.Output(
        name="cpSkyExpBackground",
        doc="Background model for a full exposure.",
        storageClass="FocalPlaneBackground",
        dimensions=("instrument", "exposure"),
    )
    outputScale = cT.Output(
        name="cpSkyExpScale",
        doc="Scale for the full exposure.",
        storageClass="PropertyList",
        dimensions=("instrument", "exposure"),
    )


class CpSkyScaleMeasureConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=CpSkyScaleMeasureConnections):
    pass


# (bg, scales)_{per exposure} = reduce(cpSkyMaskedIsr)_{over detector}
class CpSkyScaleMeasureTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Measure per-exposure scale factors and merge focal plane backgrounds.
    """
    ConfigClass = CpSkyScaleMeasureConfig
    _DefaultName = "cpSkyScaleMeasure"

    def run(self, inputBkgs):
        """Merge per-exposure scale factors and merge focal plane backgrounds.

        CZW: Write docstrings.
        """
        background = inputBkgs[0]
        for bg in inputBkgs[1:]:
            background.merge(bg)

        backgroundPixels = background.getStatsImage().getArray()
        self.log.info("Background model min/max: %f %f.  Scale %f",
                      np.min(backgroundPixels), np.max(backgroundPixels),
                      np.median(backgroundPixels))
        scale = np.median(background.getStatsImage().getArray())
        scaleMD = PropertyList()
        scaleMD.set("scale", float(scale))
        return pipeBase.Struct(
            outputBkg=background,
            outputScale=scaleMD,
        )


class CpSkySubtractBackgroundConnections(pipeBase.PipelineTaskConnections,
                                         dimensions=("instrument", "physical_filter",
                                                     "exposure", "detector")):
    inputExp = cT.Input(
        name="cpSkyMaskedIsr",
        doc="Masked post-ISR image.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    inputBkg = cT.Input(
        name="cpSkyExpBackground",
        doc="Background model for the full exposure.",
        storageClass="FocalPlaneBackground",
        dimensions=("instrument", "exposure"),
    )
    inputScale = cT.Input(
        name="cpSkyExpScale",
        doc="Scale for the full exposure.",
        storageClass="PropertyList",
        dimensions=("instrument", "exposure"),
    )

    outputBkg = cT.Output(
        name="icExpBackground",
        doc="Normalized, static background.",
        storageClass="Background",
        dimensions=("instrument", "exposure", "detector"),
    )


class CpSkySubtractBackgroundConfig(pipeBase.PipelineTaskConfig,
                                    pipelineConnections=CpSkySubtractBackgroundConnections):
    sky = pexConfig.ConfigurableField(
        target=SkyMeasurementTask,
        doc="Sky measurement",
    )


# icExpBkg_{per exposure, detector} = map(subtractBackground(bg_{exp}, scale_{exp}),
#                                         list(cpSkyMaskedIsr))
class CpSkySubtractBackgroundTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Subtract per-exposure background from individual detector masked images.
    """
    ConfigClass = CpSkySubtractBackgroundConfig
    _DefaultName = "cpSkySubtractBkg"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("sky")

    def run(self, inputExp, inputBkg, inputScale):
        """Subtract per-exposure background from individual detector masked images.

        CZW: Write docstrings.
        """
        image = inputExp.getMaskedImage()
        detector = inputExp.getDetector()
        bbox = image.getBBox()

        scale = inputScale.get('scale')
        background = inputBkg.toCcdBackground(detector, bbox)
        image -= background.getImage()
        image /= scale

        newBackground = self.sky.measureBackground(image)
        return pipeBase.Struct(
            outputBkg=newBackground
        )


class CpSkyCombineConnections(pipeBase.PipelineTaskConnections,
                              dimensions=("instrument", "physical_filter", "detector")):
    inputBkgs = cT.Input(
        name="icExpBackground",
        doc="Normalized, static background.",
        storageClass="Background",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True
    )

    outputCalib = cT.Output(
        name="skyCalib",
        doc="Averaged static background.",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "physical_filter"),
        isCalibration=True,
    )


class CpSkyCombineConfig(pipeBase.PipelineTaskConfig,
                         pipelineConnections=CpSkyCombineConnections):
    sky = pexConfig.ConfigurableField(
        target=SkyMeasurementTask,
        doc="Sky measurement",
    )


# skyCalib_{per detector} = reduce(icExpBkg)_{over exposure}
class CpSkyCombineTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Merge per-exposure measurements into a detector level calibration.
    """
    ConfigClass = CpSkyCombineConfig
    _DefaultName = "cpSkyCombine"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("sky")

    def run(self, inputBkgs):
        """Merge per-exposure measurements into a detector level calibration.

        CZW: Write docstrings.
        """
        skyCalib = self.sky.averageBackgrounds(inputBkgs)
        # This is a placeholder:
        # ETA: and it doesn't work. :(
        # CalibCombineTask().combineHeaders(inputBkgs, skyCalib, calibType='SKY')

        return pipeBase.Struct(
            outputCalib=skyCalib,
        )


class CpSkyConnections(pipeBase.PipelineTaskConnections,
                       dimensions=("instrument", "visit", "detector")):
    inputExp = cT.Input(
        name="cpSkyISR",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "detector"),
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Input camera for full focal plane backgrounds.",
        storageClass="Camera",
        dimensions=("instrument", "calibration_label"),
    )

    outputExp = cT.Output(
        name="cpSkyProc",
        doc="Output combined proposed calibration.",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "detector"),
    )


class CpSkyTaskConfig(pipeBase.PipelineTaskConfig,
                      pipelineConnections=CpSkyConnections):
    maskObjects = pexConfig.ConfigurableField(
        target=MaskObjectsTask,
        doc="Object masker to use.",
    )
    maskThresh = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="k-sigma threshold for masking pixels"
    )
    mask = pexConfig.ListField(
        dtype=str,
        default=["BAD", "SAT", "DETECTED", "NO_DATA"],
        doc="Mask planes to reject.",
    )

    largeScaleBackground = pexConfig.ConfigField(
        dtype=FocalPlaneBackgroundConfig,
        doc="Large-scale background configuration"
    )
    sky = pexConfig.ConfigurableField(
        target=SkyMeasurementTask,
        doc="Sky measurement"
    )
    background = pexConfig.ConfigField(
        dtype=BackgroundConfig,
        doc="Background config.",
    )

    # CZW:?
    detectSigma = pexConfig.Field(
        dtype=float,
        default=2.0,
        doc="Detection PSF Gaussian sigma.",
    )

    def setDefaults(self):
        self.largeScaleBackground.xSize = 256
        self.largeScaleBackground.ySize = 256


class CpSkyTask(pipeBase.PipelineTask,
                pipeBase.CmdLineTask):
    ConfigClass = CpSkyTaskConfig
    _DefaultName = "cpSky"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("maskObjects")
        self.makeSubtask("sky")

    def run(self, inputExp, camera):
        """Preprocess ISR processed exposures for combination to final SKY calibration.

        DM-22534: This isn't scaling correctly, and needs fixing.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Pre-processed sky frame data to combine.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera to use for focal-plane geometry.

        Returns
        -------
        outputExp : `lsst.afw.image.Exposure`
            Sky pre-processed data.
        """
        # As constructCalibs SkyTask.processSingleBackground
        self.maskObjects.run(inputExp, self.config.mask)

        # As constructCalibs SkyTask.measureBackground
        bgModel = FocalPlaneBackground.fromCamera(self.config.largeScaleBackground,
                                                  camera)
        bgModel.addCcd(inputExp)

        scale = np.median(bgModel.getStatsImage().getArray())

        # As constructCalibs SkyTask.scatterProcess
        # Remeasure bkg with scaled version of the FP model removed
        mi = inputExp.getMaskedImage()

        bg = bgModel.toCcdBackground(inputExp.getDetector(), mi.getBBox())
        mi -= bg.getImage()
        mi /= scale

        # As constructCalibs SkyTask.processSingle
        # Make the equivalent of the gen2 icExpBackground product.
        stats = afwMath.StatisticsControl()
        stats.setAndMask(mi.getMask().getPlaneBitMask(self.config.background.mask))
        stats.setNanSafe(True)
        ctrl = afwMath.BackgroundControl(
            self.config.background.algorithm,
            max(int(mi.getWidth() / self.config.background.xBinSize + 0.5), 1),
            max(int(mi.getHeight() / self.config.background.yBinSize + 0.5), 1),
            "REDUCE_INTERP_ORDER",
            stats,
            self.config.background.statistic
        )

        bgNew = afwMath.makeBackground(mi, ctrl)
        icExpBkg = afwMath.BackgroundList((
            bgNew,
            afwMath.stringToInterpStyle(self.config.background.algorithm),
            afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
            afwMath.ApproximateControl.UNKNOWN,
            0, 0, False
        ))

        bgExp = afwImage.makeExposure(icExpBkg[0][0].getStatsImage())

        # Return
        return pipeBase.Struct(
            outputExp=bgExp,
        )
