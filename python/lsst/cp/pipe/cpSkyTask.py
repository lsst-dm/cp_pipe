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


class CpSkyImageTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Mask the detections on the postISRCCD.

    This task maps the MaskObjectsTask across all of the initial ISR
    processed cpSkyIsr images to create cpSkyMaskedIsr products for
    all (exposure, detector) values.

    """
    ConfigClass = CpSkyImageConfig
    _DefaultName = "CpSkyImage"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("maskTask")

    def run(self, inputExp, camera):
        """Mask the detections on the postISRCCD.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            An ISR processed exposure that will have detections
            masked.
        camera : `lsst.afw.cameraGeom.Camera`
            The camera geometry for this exposure.  This is needed to
            create the background model.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``maskedExp`` : `lsst.afw.image.Exposure`
                The detection-masked version of the ``inputExp``.
            ``maskedBkg`` : `lsst.pipe.drivers.FocalPlaneBackground.`
                The partial focal plane background containing only
                this exposure/detector's worth of data.
        """
        # As constructCalibs.py SkyTask.processSingleBackground()
        # Except: check if a detector is fully masked to avoid
        # self.maskTask raising.
        currentMask = inputExp.getMask()
        badMask = currentMask.getPlaneBitMask(self.config.maskList)
        if (currentMask.getArray() & badMask).all():
            self.log.warn("All pixels are masked!")
        else:
            self.maskTask.run(inputExp, self.config.maskList)

        # As constructCalibs.py  SkyTask.measureBackground()
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
    # There are no configurable parameters here.
    pass


class CpSkyScaleMeasureTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Measure per-exposure scale factors and merge focal plane backgrounds.

    Merge all the per-detector partial backgrounds to a full focal
    plane background for each exposure, and measure the scale factor
    from that full background.
    """
    ConfigClass = CpSkyScaleMeasureConfig
    _DefaultName = "cpSkyScaleMeasure"

    def run(self, inputBkgs):
        """Merge focal plane backgrounds and measure the scale factor.

        Parameters
        ----------
        inputBkgs : `list` [`lsst.pipe.drivers.FocalPlaneBackground`]
            A list of all of the partial focal plane backgrounds, one
            from each detector in this exposure.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputBkg`` : `lsst.pipe.drivers.FocalPlaneBackground`
                The full merged background for the entire focal plane.
            ``outputScale`` : `lsst.daf.base.PropertyList`
                A metadata containing the median level of the
                background, stored in the key 'scale'.
        """
        # As constructCalibs.py SkyTask.scatterProcess()
        # Merge into the full focal plane.
        background = inputBkgs[0]
        for bg in inputBkgs[1:]:
            background.merge(bg)

        backgroundPixels = background.getStatsImage().getArray()
        self.log.info("Background model min/max: %f %f.  Scale %f",
                      np.min(backgroundPixels), np.max(backgroundPixels),
                      np.median(backgroundPixels))

        # A property list is overkill, but FocalPlaneBackground
        # doesn't have a metadata object that this can be stored in.
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
        name="cpExpBackground",
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


class CpSkySubtractBackgroundTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Subtract per-exposure background from individual detector masked images.

    The cpSkyMaskedIsr images constructed by CpSkyImageTask have the
    scaled background constructed by CpSkyScaleMeasureTask subtracted,
    and new background models are constructed for the remaining
    signal.

    The output was called `icExpBackground` in gen2, but the product
    created here has definition clashes that prevent that from being
    reused.
    """
    ConfigClass = CpSkySubtractBackgroundConfig
    _DefaultName = "cpSkySubtractBkg"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("sky")

    def run(self, inputExp, inputBkg, inputScale):
        """Subtract per-exposure background from individual detector masked images.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            The ISR processed, detection masked image.
        inputBkg : `lsst.pipe.drivers.FocalPlaneBackground.
            Full focal plane background for this exposure.
        inputScale : `lsst.daf.base.PropertyList`
            Metadata containing the scale factor.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            outputBkg : `lsst.afw.math.BackgroundList`
                Remnant sky background with the full-exposure
                component removed.
        """
        # As constructCalibs.py SkyTask.processSingle()
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
        multiple=True,
    )
    inputExps = cT.Input(
        name="cpSkyMaskedIsr",
        doc="Masked post-ISR image.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
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


class CpSkyCombineTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Merge per-exposure measurements into a detector level calibration.

    Each of the per-detector results from all input exposures are
    averaged to produce the final SKY calibration.

    As before, this is written to a skyCalib instead of a SKY to avoid
    definition classes in gen3.
    """
    ConfigClass = CpSkyCombineConfig
    _DefaultName = "cpSkyCombine"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("sky")

    def run(self, inputBkgs, inputExps):
        """Merge per-exposure measurements into a detector level calibration.

        Parameters
        ----------
        inputBkgs : `list` [`lsst.afw.math.BackgroundList`]
            Remnant backgrounds from each exposure.
        inputExps : `list` [`lsst.afw.image.Exposure`]
            The ISR processed, detection masked images.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            `outputCalib` : `lsst.afw.image.Exposure`
                The final sky calibration product.
        """
        skyCalib = self.sky.averageBackgrounds(inputBkgs)
        CalibCombineTask().combineHeaders(inputExps, skyCalib, calibType='SKY')

        return pipeBase.Struct(
            outputCalib=skyCalib,
        )
