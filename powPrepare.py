import numpy as np
import tables
import glob
import os

qScale = 0
print("scaling is qscale", str(qScale))

iTemporal = 0
iPeriodic = 1
iMesh = iPeriodic

# ------------------------- #

def getTimeSlices(baseName, directory):

    """ getTimeSlices(baseName) gets a list of files
    That will be used down the line...
    """

    filelist = glob.glob(directory + os.sep + baseName + '_integrated_*.h5')

    dumpStepNos = []

    for thisFile in filelist:
        thisDump = int(thisFile.split(os.sep)[-1].split('.')[0].split('_')[-1])
        dumpStepNos.append(thisDump)

    for i in range(len(dumpStepNos)):
        filelist[i] = os.path.join(directory, baseName + '_integrated_' + str(sorted(dumpStepNos)[i]) + '.h5')

    return filelist

# ------------------------- #

def getTimeSliceInfo(filelist, datasetname):

    """ getTimeSliceInfo() Glob for Power output and return number of files, min, max

    This should provide the number of time slices, which is used as an array dim
    It should also query the first and last to get the time (SI) and calculate z.
    z Should be present as a derived variable ultimately, but for now is not.
    """

    h5in = tables.open_file(filelist[0],'r')

    #print("Checking " + filelist[0])

    mint = h5in.root.time._v_attrs.vsTime

    try:
        minz = h5in.root._f_get_child(datasetname)._v_attrs.zbarTotal

    except:
        #print("no min z data present")
        minz = None

    h5in.close()
    h5in = tables.open_file(filelist[-1],'r')
    #print("Checking " + filelist[-1])

    maxt = h5in.root.time._v_attrs.vsTime
    try:
        maxz = h5in.root._f_get_child(datasetname)._v_attrs.zbarTotal

    except:
        #print("no max z data present")
        maxz = None

    h5in.close()

    return len(filelist), mint, maxt, minz, maxz

# ------------------------- #

def getNumSpatialPoints(filelist, datasetname):

    """ getNumSpatialPoints(filelist) get num spatial points

    What it says on the tin. Same as extent of data.
    """

    h5in = tables.open_file(filelist[0],'r')
    length = h5in.root._f_get_child(datasetname).shape[0]

    if qScale == 0:
        min_ = h5in.root.globalLimitsSI._v_attrs.vsLowerBounds
        max_ = h5in.root.globalLimitsSI._v_attrs.vsUpperBounds

    else:
        min_ = h5in.root.globalLimits._v_attrs.vsLowerBounds
        max_ = h5in.root.globalLimits._v_attrs.vsUpperBounds

    h5in.close()

    return length, min_, max_

# ------------------------- #

def getMeshType(filelist, datasetname):
    """ getNumSpatialPoints(filelist) get num spatial points

    What it says on the tin. Same as extent of data.
    """
    h5in = tables.open_file(filelist[0],'r')
    meshType = h5in.root.runInfo._v_attrs.fieldMesh
    h5in.close()

    return meshType

# ------------------------- #
# ------------------------- #

def prepare(baseName, directory):

    print("File basename specified as: " + baseName)

    outfilename = os.path.join(directory, baseName + '_integrated_all.vsh5')
    print("Will be written to: " + outfilename)

    # ------------------------- #

    if qScale == 0:
        datasetname = 'powerSI'

    else:
        datasetname = 'power'

    h5 = tables.open_file(outfilename,'w')
    filelist = getTimeSlices(baseName, directory)
    numTimes, minZT, maxZT, minZZ, maxZZ = getTimeSliceInfo(filelist,datasetname)
    numSpatialPoints, minS, maxS = getNumSpatialPoints(filelist,datasetname)
    lenz2 = maxS - minS
    deltaz2 = (maxS - minS) / numSpatialPoints
    iMesh = getMeshType(filelist,datasetname)
    sumData = np.zeros(numTimes)
    peakData = np.zeros(numTimes)
    powAv = np.zeros(numTimes)

    h5.create_group('/','gridZ_SI','')
    numCells = np.array((np.int(numSpatialPoints)-1,np.int(numTimes)-1))
    h5.root.gridZ_SI._v_attrs.vsLowerBounds = np.array((np.double(minS),np.double(minZT)))
    h5.root.gridZ_SI._v_attrs.vsStartCell = np.array((np.int(0),np.int(0)))
    h5.root.gridZ_SI._v_attrs.vsUpperBounds = np.array((np.double(maxS),np.double(maxZT)))
    h5.root.gridZ_SI._v_attrs.vsNumCells = np.array(numCells)
    h5.root.gridZ_SI._v_attrs.vsKind = "uniform"
    h5.root.gridZ_SI._v_attrs.vsType = "mesh"
    h5.root.gridZ_SI._v_attrs.vsCentering = "nodal"
    h5.root.gridZ_SI._v_attrs.vsAxisLabels = "ct-z,z"


    if minZZ is not None:
        h5.create_group('/','gridZScaled','')
        h5.root.gridZScaled._v_attrs.vsLowerBounds = np.array((np.double(minS),np.double(minZZ)))
        h5.root.gridZScaled._v_attrs.vsStartCell = np.array((np.int(0),np.int(0)))
        h5.root.gridZScaled._v_attrs.vsUpperBounds = np.array((np.double(maxS),np.double(maxZZ)))
        h5.root.gridZScaled._v_attrs.vsNumCells = np.array(numCells)
        h5.root.gridZScaled._v_attrs.vsKind = "uniform"
        h5.root.gridZScaled._v_attrs.vsType = "mesh"
        h5.root.gridZScaled._v_attrs.vsAxisLabels = "Z2bar,Zbar"
        h5.root.gridZScaled._v_attrs.vsCentering = "nodal"


    fieldData = np.zeros((numSpatialPoints, numTimes))
    fieldNormData = np.zeros((numSpatialPoints, numTimes))
    zData = np.zeros((numTimes))
    fieldCount = 0



    # so want to do this for power and power_SI...???
    # since datasetname = power here...

    for slice in filelist:

        h5in = tables.open_file(slice,'r')
        fieldData[:, fieldCount] = h5in.root._f_get_child(datasetname).read()
        sumData[fieldCount] = np.trapz(h5in.root._f_get_child(datasetname).read(), None, deltaz2) 
        powAv[fieldCount] = sumData[fieldCount] / lenz2
        peakData[fieldCount] = np.max(h5in.root._f_get_child(datasetname).read())

        if peakData[fieldCount] != 0:
            fieldNormData[:, fieldCount] = h5in.root._f_get_child(datasetname).read()/peakData[fieldCount]

        else:
            fieldNormData[:, fieldCount] = h5in.root._f_get_child(datasetname).read()

        # for including drifts
        if qScale == 0:
            zData[fieldCount] = h5in.root._f_get_child("power")._v_attrs.zTotal 

        else:
            zData[fieldCount] = h5in.root._f_get_child("power")._v_attrs.zbarTotal 
        # for no drifts
        # zData[fieldCount] = h5in.root._f_get_child("power")._v_attrs.zInter
        h5in.close()
        fieldCount += 1

    c0 = 2.998E8

    if qScale == 0:
        sumData = sumData / c0   # power = integral over t, not z = ct!!

    # Creating SI power and normalized power datasets...

    if qScale == 0:
        h5.create_array('/','power_SI',fieldData)
        h5.create_array('/','power_SI_Norm',fieldNormData)
    
        for fieldname in ['power_SI','power_SI_Norm']:
            h5.root._v_children[fieldname]._v_attrs.vsMesh = "gridTPowEv"
            h5.root._v_children[fieldname]._v_attrs.vsType = "variable"

    else:
        h5.create_array('/','power_scaled',fieldData)
        h5.create_array('/','power_scaled_Norm',fieldNormData)

        for fieldname in ['power_scaled','power_scaled_Norm']:
            h5.root._v_children[fieldname]._v_attrs.vsMesh = "gridTPowEv"
            h5.root._v_children[fieldname]._v_attrs.vsType = "variable"



    h5.create_group('/','time','')
    h5.root.time._v_attrs.vsType = "time"
    h5.root.time._v_attrs.vsTime = 0.
    h5.root.time._v_attrs.vsStep = 0


    h5.create_array('/','Energy',sumData)
    h5.root.Energy._v_attrs.vsMesh = 'zSeries'
    h5.root.Energy._v_attrs.vsType = 'variable'

    if qScale == 0:
        h5.root.Energy._v_attrs.vsAxisLabels = 'z (m), Energy (J)'

    else:
        h5.root.Energy._v_attrs.vsAxisLabels = 'zbar, Energy (arb. units)'


    h5.create_array('/','PeakPower',peakData)
    h5.root.PeakPower._v_attrs.vsMesh = 'zSeries'
    h5.root.PeakPower._v_attrs.vsType = 'variable'

    if qScale == 0:
        h5.root.PeakPower._v_attrs.vsAxisLabels = 'z (m), Pk Pow (W)'

    else:
        h5.root.PeakPower._v_attrs.vsAxisLabels = 'zbar, Pk Pow (scaled)'


    if iMesh == iPeriodic:

        h5.create_array('/','Power',powAv)
        h5.root.Power._v_attrs.vsMesh = 'zSeries'
        h5.root.Power._v_attrs.vsType = 'variable'

        if qScale == 0:
            h5.root.Power._v_attrs.vsAxisLabels = 'z (m), Pow (W)'

        else:
            h5.root.Power._v_attrs.vsAxisLabels = 'zbar, Power (scaled)'

    h5.create_array('/','zSeries', zData)
    h5.root.zSeries._v_attrs.vsKind = 'structured'
    h5.root.zSeries._v_attrs.vsType = 'mesh'
    h5.root.zSeries._v_attrs.vsStartCell = 0
    h5.root.zSeries._v_attrs.vsLowerBounds = zData[0]
    h5.root.zSeries._v_attrs.vsUpperBounds = zData[-1]

    if qScale == 0:
        h5.root.zSeries._v_attrs.vsAxisLabels = "z (m)"

    else:
        h5.root.zSeries._v_attrs.vsAxisLabels = "zbar"

    z2Data = np.linspace(np.double(minS), np.double(maxS), np.int(numSpatialPoints))
    XG, YG = np.meshgrid(z2Data, zData)
    comb = np.zeros((np.int(numSpatialPoints), numTimes, 2))

    comb[:,:,0] = XG.T
    comb[:,:,1] = YG.T

    h5.create_array('/','gridTPowEv',comb)

    h5.root.gridTPowEv._v_attrs.vsKind="structured"
    h5.root.gridTPowEv._v_attrs.vsType="mesh"
    h5.root.gridTPowEv._v_attrs.vsCentering="nodal"
    h5.root.gridTPowEv._v_attrs.vsLowerBounds=np.array((np.double(zData[0]),np.double(z2Data[0])))
    h5.root.gridTPowEv._v_attrs.vsUpperBounds=np.array((np.double(zData[-1]),np.double(z2Data[-1])))

    if qScale == 0:
        h5.root.gridTPowEv._v_attrs.vsAxisLabels = "ct-z, z"
        
    else:
        h5.root.gridTPowEv._v_attrs.vsAxisLabels = "z2, zbar"

    h5in = tables.open_file(filelist[-1])
    h5in.root.runInfo._f_copy(h5.root)
    h5in.close()
    h5.close()