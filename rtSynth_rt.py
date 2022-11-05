'''
    steps:
        conver the new dicom to nii
        align the nii to cfg.templateFunctionalVolume_converted
        apply mask 
        load clf
        get morphing parameter
    '''
"""-----------------------------------------------------------------------------

    sample.py (Last Updated: 05/26/2020)

    The purpose of this script is to actually to run the sample project.
    Specifically, it will initiate a file watcher that searches for incoming dicom
    files, do some sort of analysis based on the dicom file that's been received,
    and then output the answer.

    The purpose of this *particular* script is to demonstrated how you can use the
    various scripts, functions, etc. we have developed for your use! The functions
    we will reference live in 'rt-cloud/rtCommon/'.

    Finally, this script is called from 'projectMain.py', which is called from
    'run-projectInterface.sh'.

    -----------------------------------------------------------------------------"""
verbose = False
useInitWatch = False

if verbose:
    # print a short introduction on the internet window
    print(""
        "-----------------------------------------------------------------------------\n"
        "The purpose of this sample project is to demonstrate different ways you can\n"
        "implement functions, structures, etc. that we have developed for your use.\n"
        "You will find some comments printed on this html browser. However, if you want\n"
        "more information about how things work please take a look at ‘sample.py’.\n"
        "Good luck!\n"
        "-----------------------------------------------------------------------------")
# import important modules
import os,time
import sys
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/')
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/')
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/')
import argparse
import warnings
import numpy as np
import nibabel as nib
import scipy.io as sio
from cfg_loading import mkdir,cfg_loading
from subprocess import call
import joblib
import pandas as pd
from scipy.stats import zscore
if verbose:
    print(''
        '|||||||||||||||||||||||||||| IGNORE THIS WARNING ||||||||||||||||||||||||||||')
with warnings.catch_warnings():
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning)
    from nibabel.nicom import dicomreaders

if verbose:
    print(''
        '|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')

# obtain full path for current directory: '.../rt-cloud/projects/sample'
currPath = os.path.dirname(os.path.realpath(__file__))
# obtain full path for root directory: '.../rt-cloud'
rootPath = os.path.dirname(os.path.dirname(currPath))

# add the path for the root directory to your python path so that you can import
#   project modules from rt-cloud
sys.path.append(rootPath)
from rtCommon.utils import loadConfigFile, stringPartialFormat
from rtCommon.clientInterface import ClientInterface
from rtCommon.imageHandling import readRetryDicomFromDataInterface, convertDicomImgToNifti
from rtCommon.dataInterface import DataInterface #added by QL
from recognition_dataAnalysisFunctions import normalize,classifierProb


# obtain the full path for the configuration toml file
# defaultConfig = os.path.join(currPath, 'conf/sample.toml')
defaultConfig = '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/projects/rtSynth_rt/conf/'+"sub001.ses3.toml"


def doRuns(cfg, dataInterface, subjInterface, webInterface):
    """
        This function is called by 'main()' below. Here, we use the 'fileInterface'
        to read in dicoms (presumably from the scanner, but here it's from a folder
        with previously collected dicom files), doing some sort of analysis in the
        cloud, and then sending the info to the web browser.

        INPUT:
            [1] cfg (configuration file with important variables)
            [2] fileInterface (this will allow a script from the cloud to access files
                from the stimulus computer, which receives dicom files directly
                from the Siemens console computer)
            [3] projectComm (communication pipe to talk with projectInterface)
        OUTPUT:
            None.

        This is the main function that is called when you run 'sample.py'.
        Here, you will set up an important argument parser (mostly provided by
        the toml configuration file), initiate the class fileInterface, and then
        call the function 'doRuns' to actually start doing the experiment.
    """

    # variables we'll use throughout
    scanNum = cfg.scanNum[0]
    runNum = cfg.runNum[0]

    print(f"Doing run {runNum}, scan {scanNum}")
    print(f"cfg.dicomDir={cfg.dicomDir}")
    """
        Before we get ahead of ourselves, we need to make sure that the necessary file
            types are allowed (meaning, we are able to read them in)... in this example,
            at the very least we need to have access to dicom and txt file types.
        use the function 'allowedFileTypes' in 'fileClient.py' to check this!
        If allowedTypes doesn't include the file types we need to use then the 
            file service (scannerDataService) running at the control room computer will
            need to be restarted with the correct list of allowed types provided.

        INPUT: None
        OUTPUT:
            [1] allowedFileTypes (list of allowed file types)
    """
    allowedFileTypes = dataInterface.getAllowedFileTypes()
    if verbose:
        print(""
        "-----------------------------------------------------------------------------\n"
        "Before continuing, we need to make sure that dicoms are allowed. To verify\n"
        "this, use the 'allowedFileTypes'.\n"
        "Allowed file types: %s" %allowedFileTypes)

    # obtain the path for the directory where the subject's dicoms live
    # if cfg.isSynthetic:
    #     cfg.dicomDir = cfg.imgDir
    # else:
    #     subj_imgDir = "{}.{}.{}".format(cfg.datestr, cfg.subjectName, cfg.subjectName)
    #     cfg.dicomDir = os.path.join(cfg.imgDir, subj_imgDir)
    if verbose:
        print("Location of the subject's dicoms: \n" + cfg.dicomDir + "\n"
        "-----------------------------------------------------------------------------")

    #  If a dicomNamePattern is supplied in the config file, such as
    #  "001_{SCAN:06d}_{TR:06d}.dcm", then call stringPartialFormat() to 
    #  set the SCAN number for the series of Dicoms we will be streaming.
    dicomScanNamePattern = stringPartialFormat(cfg.dicomNamePattern, 'SCAN', scanNum)
    print(f"dicomScanNamePattern={dicomScanNamePattern}")

    """
        There are several ways to receive Dicom data from the control room computer:
        1. Using `initWatch()` and 'watchFile()` commands of dataInterface or the
            helper function `readRetryDicomFromDataInterface()` which calls watchFile()
            internally.
        2. Using the streaming functions with `initScannerStream()` and `getImageData(stream)`
            which are also part of the dataInterface.
    """
    if useInitWatch is True:
        """
            Initialize a watch for the entire dicom folder using the function 'initWatch'
            of the dataInterface. (Later we will use watchFile() to look for a specific dicom)
            INPUT:
                [1] cfg.dicomDir (where the subject's dicom files live)
                [2] cfg.dicomNamePattern (the naming pattern of dicom files)
                [3] cfg.minExpectedDicomSize (a check on size to make sure we don't
                        accidentally grab a dicom before it's fully acquired)
        """
        if verbose:
            print("• initalize a watch for the dicoms using 'initWatch'")
            print(f"cfg.dicom_dir={cfg.dicom_dir}, cfg.dicomNamePattern={cfg.dicomNamePattern}, \
                cfg.minExpectedDicomSize={cfg.minExpectedDicomSize}")
        dataInterface.initWatch(cfg.dicomDir, dicomScanNamePattern, cfg.minExpectedDicomSize)
    else:  # use Stream functions
        """
            Initialize a Dicom stream by indicating the directory and dicom file pattern that
            will be streamed.

            INPUTs to initScannerStream():
                [1] cfg.dicomDir (where the subject's dicom files live)
                [2] dicomScanNamePattern (the naming pattern of dicom files)
                [3] cfg.minExpectedDicomSize (a check on size to make sure we don't
                        accidentally grab a dicom before it's fully acquired)
        """
        if verbose:
            print(f"cfg.dicomDir={cfg.dicomDir}, dicomScanNamePattern={dicomScanNamePattern}, cfg.minExpectedDicomSize={cfg.minExpectedDicomSize})")
            print(f"cfg.dicom_dir={cfg.dicom_dir}, cfg.dicomNamePattern={cfg.dicomNamePattern}, \
                cfg.minExpectedDicomSize={cfg.minExpectedDicomSize}")
        print(f"cfg.minExpectedDicomSize={cfg.minExpectedDicomSize}")
        streamId = dataInterface.initScannerStream(cfg.dicomDir, 
                                                dicomScanNamePattern,
                                                cfg.minExpectedDicomSize)


    """
        We will use the function plotDataPoint in webInterface whenever we
        want to send values to the web browser so that they can be plotted in the
        --Data Plots-- tab. 
        However at the start of a run we will want to clear the plot, and we can use
        clearRunPlot(runId), or clearAllPlots() also in the webInterface object.
    """
    if verbose:
        print("• clear any pre-existing plot for this run using 'clearRunPlot(runNum)'")
    webInterface.clearRunPlot(runNum)

    if verbose:
        print(""
        "-----------------------------------------------------------------------------\n"
        "In this sample project, we will retrieve the dicom file for a given TR and\n"
        "then convert the dicom file to a nifti object. **IMPORTANT: In this sample\n"
        "we won't care about the exact location of voxel data (we're only going to\n"
        "indiscriminately get the average activation value for all voxels). This\n"
        "actually isn't something you want to actually do but we'll go through the\n"
        "to get the data in the appropriate nifti format in the advanced sample\n"
        "project (amygActivation).** We are doing things in this way because it is the simplest way\n"
        "we can highlight the functionality of rt-cloud, which is the purpose of\n"
        "this sample project.\n"
        ".............................................................................\n"
        "NOTE: We will use the function readRetryDicomFromDataInterface() to retrieve\n"
        "specific dicom files from the subject's dicom folder. This function calls\n"
        "'dataInterface.watchFile' to look for the next dicom from the scanner.\n"
        "Since we're using previously collected dicom data, this functionality is\n"
        "not particularly relevant for this sample project but it is very important\n"
        "when running real-time experiments.\n"
        "-----------------------------------------------------------------------------\n")

    tmp_dir=f"{cfg.tmp_folder}{time.time()}/" ; mkdir(tmp_dir)

    # forceGreedy = ""
    # if forceGreedy=="forceGreedy":
        # mask = np.load(f"{cfg.chosenMask_using}")
    # else:
    mask = np.load(f"{cfg.chosenMask}")

    imcodeDict={
        'A': 'bed',
        'B': 'chair',
        'C': 'table',
        'D': 'bench'}
    """ 不同batch的命名方法，下一步是收集batch2的数据
        batch1
            present bed, driving for chair;
            morphing from bed to table and bench;
            driving clf is mean of chair_table and chair_bench clf prob

            same :
            present A, driving for B;
            morphing from A to C and D;
            driving clf is mean of BC and BD clf prob
        batch2
            present C, driving for D;
            morphing from C to A and B;
            driving clf is mean of DA and DB clf prob
        batch3
            present B, driving for A;
            morphing from B to C and D;
            driving clf is mean of AC and AD clf prob
        batch4
            present D, driving for C;
            morphing from D to A and B;
            driving clf is mean of CA and CB clf prob
        
        """

    
    # next subject sub016 is batch2.
    if cfg.batch=='batch1':
        BC_clf=joblib.load(cfg.usingModel_dir +'benchchair_chairtable.joblib') # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
        BD_clf=joblib.load(cfg.usingModel_dir +'bedchair_chairbench.joblib') # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
    elif cfg.batch=='batch2':
        DA_clf=joblib.load(cfg.usingModel_dir +'benchtable_benchbed.joblib') # benchtable_benchbed benchchair_benchbed bedtable_bedbench bedchair_bedbench
        DB_clf=joblib.load(cfg.usingModel_dir +'benchtable_benchchair.joblib') # benchtable_benchchair bedbench_benchchair chairtable_chairbench bedchair_chairbench
    logTimes=[]
    # where the morphParams are saved
    # output_textFilename = f'{cfg.feedback_dir}probs_{scanNum}.txt'
    output_matFilename = os.path.join(f'{cfg.feedback_dir}probs_{scanNum}.mat')
    
    num_total_trials=12
    num_total_TRs = int((num_total_trials*28+12)/2) + 8  # number of TRs to use for example 1
    # morphParams = np.zeros((num_total_TRs, 1))
    probs=[]
    maskedData=0
    timeout_file = 5 # small number because of demo, can increase for real-time
    processedTime=[] # for each this_TR (in dicom folder TR start from 1)
    for this_TR in np.arange(1,num_total_TRs):
        print(f"milgramTR_ID={this_TR}")
        # declare variables that are needed to use 'readRetryDicomFromFileInterface'
        
        dicomFilename = dicomScanNamePattern.format(TR=this_TR)
        
        if useInitWatch is True:
            """
                Use 'readRetryDicomFromDataInterface' in 'imageHandling.py' to wait for dicom
                    files to be written by the scanner (uses 'watchFile' internally) and then
                    reading the dicom file once it is available.
                INPUT:
                    [1] dataInterface (allows a cloud script to access files from the
                        control room computer)
                    [2] filename (the dicom file we're watching for and want to load)
                    [3] timeout (time spent waiting for a file before timing out)
                OUTPUT:
                    [1] dicomData (with class 'pydicom.dataset.FileDataset')
            """
            print(f'Processing TR {this_TR}')
            if verbose:
                print("• use 'readRetryDicomFromDataInterface' to read dicom file for",
                    "TR %d, %s" %(this_TR, dicomFilename))
            # dicomData = readRetryDicomFromDataInterface(dataInterface, dicomFilename, timeout_file)  
            # print(f"{cfg.dicom_dir}/{dicomFilename}")
            # while True:
            #     if os.path.exists(f"{cfg.dicom_dir}/{dicomFilename}"):
            #         print(f"found {cfg.dicom_dir}/{dicomFilename}")
            #         break
            #     else:
            #         print(f"not found {cfg.dicom_dir}/{dicomFilename}")
                
            dicomData = readRetryDicomFromDataInterface(dataInterface, dicomFilename, timeout_file)  
        else:  # use Stream functions
            """
            Use dataInterface.getImageData(streamId) to query a stream, waiting for a 
                dicom file to be written by the scanner and then reading the dicom file
                once it is available.
            INPUT:
                [1] dataInterface (allows a cloud script to access files from the
                    control room computer)
                [2] streamId - from initScannerStream() called above
                [3] TR number - the image volume number to retrieve
                [3] timeout (time spent waiting for a file before timing out)
            OUTPUT:
                [1] dicomData (with class 'pydicom.dataset.FileDataset')
            """
            print(f'Processing TR {this_TR}')
            if verbose:
                print("• use dataInterface.getImageData() to read dicom file for"
                    "TR %d, %s" %(this_TR, dicomFilename))
            print(f"{cfg.dicom_dir}/{dicomFilename}")
            # while True:
            #     if os.path.exists(f"{cfg.dicom_dir}/{dicomFilename}"):
            #         print(f"found {cfg.dicom_dir}/{dicomFilename}")
            #         time.sleep(0.1) # 100ms sleep
            #         break
            #     # else:
            #     #     print(f"not found {cfg.dicom_dir}/{dicomFilename}")
            timeout_file=5
            dicomData = dataInterface.getImageData(streamId, int(this_TR), timeout_file)
            logTime=time.time() # rtSynth_rt成功获得dicom file的时候
            processing_start_time=time.time()
            logTimes.append(logTime)

        # processing_start_time=time.time()
        if dicomData is None:
            print('Error: getImageData returned None')
            return         
        dicomData.convert_pixel_data()

        if verbose:
            print("| convert dicom data into a nifti object")
        niftiObject = dicomreaders.mosaic_to_nii(dicomData)
        # print(f"niftiObject={niftiObject}")

        # save(f"{tmp_dir}niftiObject")
        # niiFileName=f"{tmp_dir}{fileName.split('/')[-1].split('.')[0]}.nii"
        niiFileName= tmp_dir+cfg.dicomNamePattern.format(SCAN=scanNum,TR=this_TR).split('.')[0]
        print(f"niiFileName={niiFileName}.nii")
        nib.save(niftiObject, f"{niiFileName}.nii")  
        # align -in f"{tmp_dir}niftiObject" -ref cfg.templateFunctionalVolume_converted -out f"{tmp_dir}niftiObject"
        # 由于遇到了这个bug：Input: A-P R-L I-S
            # Base:  R-L P-A I-S
            # ** FATAL ERROR: perhaps you could make your datasets match?

        # 因此使用3dresample来处理这个bug
        command=f"3dresample \
            -master {cfg.templateFunctionalVolume_converted} \
            -prefix {niiFileName}_reorient.nii \
            -input {niiFileName}.nii"
        print(command)
        call(command,shell=True)

        command = f"3dvolreg \
                -base {cfg.templateFunctionalVolume_converted} \
                -prefix  {niiFileName}_aligned.nii \
                {niiFileName}_reorient.nii"

        print(command)
        call(command,shell=True)

        niftiObject = nib.load(f"{niiFileName}_aligned.nii")
        nift_data = niftiObject.get_fdata()
        
        curr_volume = np.expand_dims(nift_data[mask==1], axis=0)
        maskedData=curr_volume if this_TR==1 else np.concatenate((maskedData,curr_volume),axis=0)
        _maskedData = normalize(maskedData)

        print(f"_maskedData.shape={_maskedData.shape}")

        X = np.expand_dims(_maskedData[-1], axis=0)
        
        # imcodeDict={
        # 'A': 'bed',
        # 'B': 'chair',
        # 'C': 'table',
        # 'D': 'bench'}

        def get_prob(showingImage="A", drivingTarget="B", otherAxis1="C", otherAxis2="D", 
                    drivingClf1='BC_clf', drivingClf2='BD_clf',
                    X="X"): # X is the current volume
            Y = imcodeDict[drivingTarget] # is chair when batch1
            print(f"classifierProb({drivingTarget}{otherAxis1}_clf,X,Y)={classifierProb(drivingClf1,X,Y)}")
            print(f"classifierProb({drivingTarget}{otherAxis2}_clf,X,Y)={classifierProb(drivingClf2,X,Y)}")
            prob1 = classifierProb(drivingClf1,X,Y)[0]
            prob2 = classifierProb(drivingClf2,X,Y)[0]
            print(f"{drivingTarget}{otherAxis1}_{drivingTarget}_prob={prob1}")
            print(f"{drivingTarget}{otherAxis1}_{drivingTarget}_prob={prob2}")
            prob = float((prob1+prob2)/2)
            print(f"{drivingTarget}_prob={prob}")           
            print(f"| {drivingTarget}_prob for TR {this_TR} is {prob}")
            return prob

        if cfg.batch=='batch1':
            oldVersion=False
            if oldVersion==True:
                Y = 'chair'
                print(f"classifierProb(BC_clf,X,Y)={classifierProb(BC_clf,X,Y)}")
                print(f"classifierProb(BD_clf,X,Y)={classifierProb(BD_clf,X,Y)}")
                BC_B_prob = classifierProb(BC_clf,X,Y)[0]
                BD_B_prob = classifierProb(BD_clf,X,Y)[0]
                print(f"BC_B_prob={BC_B_prob}")
                print(f"BD_B_prob={BD_B_prob}")
                B_prob = float((BC_B_prob+BD_B_prob)/2)
                print(f"B_prob={B_prob}")           
                print("| B_prob for TR %d is %f" %(this_TR, B_prob))
                prob = B_prob
            else:
                prob = get_prob(showingImage="A", drivingTarget="B", otherAxis1="C", otherAxis2="D", drivingClf1=BC_clf, drivingClf2=BD_clf, X=X) # X is the current volume

            probs.append(prob)
        elif cfg.batch=='batch2':
            prob = get_prob(showingImage="C", drivingTarget="D", otherAxis1="A", otherAxis2="B", drivingClf1=DA_clf, drivingClf2=DB_clf, X=X) # X is the current volume


        # use 'sendResultToWeb' from 'projectUtils.py' to send the result to the
        #   web browser to be plotted in the --Data Plots-- tab.
        

        if verbose:
            print("| send result to the presentation computer for provide subject feedback")
        
        subjInterface.setResult(runNum, int(this_TR), prob)

        if verbose:
            print("| send result to the web, plotted in the 'Data Plots' tab")
        webInterface.plotDataPoint(runNum, int(this_TR), prob)

        # save the activations value info into a vector that can be saved later
        # morphParams[this_TR] = morphParam

        # dataInterface.putFile(output_textFilename,str(probs))
        np.save(f'{cfg.feedback_dir}probs_{scanNum}',probs)
        processing_end_time=time.time()
        print(f"{processing_end_time-processing_start_time} s passes when processing")
        processedTime.append(processing_end_time-processing_start_time)
        np.save(f'{cfg.feedback_dir}processedTime_scan{scanNum}',processedTime)
        np.save(f'{cfg.feedback_dir}logTimes_{scanNum}',np.asarray(logTimes))

    # create the full path filename of where we want to save the activation values vector
    #   we're going to save things as .txt and .mat files

    # use 'putTextFile' from 'fileClient.py' to save the .txt file
    #   INPUT:
    #       [1] filename (full path!)
    #       [2] data (that you want to write into the file)

    # if verbose:
    #     print(""
    #     "-----------------------------------------------------------------------------\n"
    #     "• save activation value as a text file to tmp folder")
    # dataInterface.putFile(output_textFilename,str(probs))

    # use sio.save mat from scipy to save the matlab file
    if verbose:
        print("• save activation value as a matlab file to tmp folder")
    sio.savemat(output_matFilename,{'value':probs})

    if verbose:
        print(""
        "-----------------------------------------------------------------------------\n"
        "REAL-TIME EXPERIMENT COMPLETE!")

    return


def main(argv=None):
    global verbose, useInitWatch
    """
    This is the main function that is called when you run 'sample.py'.

    Here, you will load the configuration settings specified in the toml configuration
    file, initiate the clientInterface for communication with the projectServer (via
    its sub-interfaces: dataInterface, subjInterface, and webInterface). Ant then call
    the function 'doRuns' to actually start doing the experiment.
    """

    # Some generally recommended arguments to parse for all experiment scripts
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                           help='experiment config file (.json or .toml)')
    argParser.add_argument('--runs', '-r', default=None, type=str,
                           help='Comma separated list of run numbers')
    argParser.add_argument('--scans', '-s', default=None, type=str,
                           help='Comma separated list of scan number')
    argParser.add_argument('--yesToPrompts', '-y', default=False, action='store_true',
                           help='automatically answer tyes to any prompts')

    # Some additional parameters only used for this sample project
    argParser.add_argument('--useInitWatch', '-w', default=False, action='store_true',
                           help='use initWatch() functions instead of stream functions')
    argParser.add_argument('--Verbose', '-v', default=False, action='store_true',
                           help='print verbose output')


    args = argParser.parse_args(argv)

    useInitWatch = args.useInitWatch
    verbose = args.Verbose

    # load the experiment configuration file
    print(f"rtSynth_rt: args.config={args.config}")
    # if trying:
        # cfg = cfg_loading(args.config,trying="trying")
    # else:
    cfg = cfg_loading(args.config)


    # override config file run and scan values if specified
    if args.runs is not None:
        print("runs: ", args.runs)
        cfg.runNum = [int(x) for x in args.runs.split(',')]
    if args.scans is not None:
        print("scans: ", args.scans)
        cfg.ScanNum = [int(x) for x in args.scans.split(',')]

    # Initialize the RPC connection to the projectInterface.
    # This will give us a dataInterface for retrieving files,
    # a subjectInterface for giving feedback, and a webInterface
    # for updating what is displayed on the experimenter's webpage.
    clientInterfaces = ClientInterface(yesToPrompts=args.yesToPrompts)
    #dataInterface = clientInterfaces.dataInterface
    subjInterface = clientInterfaces.subjInterface
    webInterface  = clientInterfaces.webInterface

    ## Added by QL
    allowedDirs = ['*'] #['/gpfs/milgram/pi/turk-browne/projects/rt-cloud/projects/sample/dicomDir/20190219.0219191_faceMatching.0219191_faceMatching','/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/sample', '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/sample/dicomDir']
    allowedFileTypes = ['*'] #['.txt', '.dcm']
    dataInterface = DataInterface(dataRemote=False,allowedDirs=allowedDirs,allowedFileTypes=allowedFileTypes) # Create an instance of local datainterface

    # Also try the placeholder for bidsInterface (an upcoming feature)
    bidsInterface = clientInterfaces.bidsInterface
    # res = bidsInterface.echo("test")
    # print(res)

    # obtain paths for important directories (e.g. location of dicom files)
    # if cfg.imgDir is None:
    #     cfg.imgDir = os.path.join(currPath, 'dicomDir')
    # cfg.codeDir = currPath

    # now that we have the necessary variables, call the function 'doRuns' in order
    #   to actually start reading dicoms and doing your analyses of interest!
    #   INPUT:
    #       [1] cfg (configuration file with important variables)
    #       [2] dataInterface (this will allow a script from the cloud to access files
    #            from the stimulus computer that receives dicoms from the Siemens
    #            console computer)
    #       [3] subjInterface - this allows sending feedback (e.g. classification results)
    #            to a subjectService running on the presentation computer to provide
    #            feedback to the subject (and optionally get their response).
    #       [4] webInterface - this allows updating information on the experimenter webpage.
    #            For example to plot data points, or update status messages.
    doRuns(cfg, dataInterface, subjInterface, webInterface)
    return 0

if __name__ == "__main__":
    """
    If 'sample.py' is invoked as a program, then actually go through all of the portions
    of this script. This statement is not satisfied if functions are called from another
    script using "from sample.py import FUNCTION"
    """
    main()
    sys.exit(0)

