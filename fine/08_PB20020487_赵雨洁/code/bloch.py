import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import proj3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.patches import FancyArrowPatch
import numpy as np
from numbers import Number
import scipy.integrate as integrate
from scipy.stats import norm
import os.path
import argparse
import yaml
import FFMPEGwriter
from bloch_draw import plotFrame3D,getEventFrames,getText

gyro = 42577.	

#布洛克方程
def bloch(M, t, Meq, w, w1, T1, T2):
    dMdt = np.zeros_like(M)
    dMdt[0] = -M[0]/T2+M[1]*w+M[2]*w1.real
    dMdt[1] = -M[0]*w-M[1]/T2+M[2]*w1.imag
    dMdt[2] = -M[0]*w1.real-M[1]*w1.imag+(Meq-M[2])/T1
    return dMdt

#计算脉冲造成的影响
def applyPulseSeq(config, Meq, M0, w, T1, T2, pos0, v, D):
    M = np.zeros([len(config['t']), 3])
    M[0] = M0 
    pos = np.tile(pos0, [len(config['t']), 1]) 
    for rep in range(-config['nDummies'], config['nTR']): 
        TRstartFrame = rep * config['nFramesPerTR']
        for i, event in enumerate(config['events']):
            firstFrame, lastFrame = getEventFrames(config, i)
            firstFrame += TRstartFrame
            lastFrame += TRstartFrame

            M0 = M[firstFrame]
            wg = w  
            wg += 2*np.pi*gyro*event['Gx']*pos[firstFrame, 0]/1000
            wg += 2*np.pi*gyro*event['Gy']*pos[firstFrame, 1]/1000 
            wg += 2*np.pi*gyro*event['Gz']*pos[firstFrame, 2]/1000 
            #计算章动频率
            w1 = event['w1'] * np.exp(1j * np.radians(event['phase']))#计算进动频率，角度换成弧度
            t = config['t'][firstFrame:lastFrame+1]
            M[firstFrame:lastFrame+1] = integrate.odeint(bloch, M0, t, args=(Meq, wg, w1, T1, T2)) 
            #数值求解
    return np.concatenate((M, pos),1).transpose()


def simulateComponent(config, component, Meq, M0=None, pos=None):
    if not M0:
        M0 = [0, 0, Meq] #设定默认值
    if not pos:
        pos = [0, 0, 0]#设定默认值
    v = [component['vx'], component['vy'], component['vz']]
    D = [component['Dx'], component['Dy'], component['Dz']]
    isochromats = [(2*i+1-config['nIsochromats'])/2*config['isochromatStep']+component['CS'] for i in range(0, config['nIsochromats'])]
    comp = np.empty((config['nIsochromats'],6,len(config['t'])))
  
    for m, isochromat in enumerate(isochromats):
        w = config['w0']*isochromat*1e-6  
        comp[m,:,:] = applyPulseSeq(config, Meq, M0, w, component['T1'], component['T2'], pos, v, D)
    return comp


def roundEventTime(time):
    return np.round(time, decimals=6) #四舍五入时间使其为帧的整数倍

#把事件刻到时间坐标
def addEventsToTimeVector(t, pulseSeq):
    t = list(t)
    for event in pulseSeq:
        t.append(event['t'])
    return np.unique(roundEventTime(np.array(t)))


def checkPulseSeq(config):
    if 'pulseSeq' not in config:
        config['pulseSeq'] = []
    allowedKeys = ['t',  'dur', 'FA', 'B1', 'phase']
    for event in config['pulseSeq']:
        
            if 'FA' in event:

                event['B1'] = np.array([event['FA']/(event['dur'] * 360 * gyro * 1e-6)])
               
            event['w1'] = [2 * np.pi * gyro * B1 * 1e-6 for B1 in event['B1']] # 拉莫频率 kRad / s
            event['RFtext'] = str(int(abs(event['FA'])))+u'\N{DEGREE SIGN}'+'-pulse'

    # 把脉冲分为多个事件
    config['pulseSeq'] = sorted(config['pulseSeq'], key=lambda event: event['t'])
    config['separatedPulseSeq'] = []
    for event in config['pulseSeq']:
        arrLengths = len(event['w1']) 
        for i, t in enumerate(np.linspace(event['t'], event['t'] + event['dur'], arrLengths, endpoint=False)):
                subDur = event['dur'] / arrLengths
                subEvent = {'t': t, 'dur': subDur}             
                for key in ['w1', 'Gx', 'Gy', 'Gz', 'phase', 'RFtext']:
                    if key in event:
                        if type(event[key]) is list:
                            if i < len(event[key]):
                                subEvent[key] = event[key][i]
                            else:
                                raise Exception('Length of {} does not match other event properties'.format(key))
                        else:
                            subEvent[key] = event[key]
                        if key in ['Gx', 'Gy', 'Gz']:
                            subEvent['{}text'.format(key)] = '{}: {:2.0f} mT/m'.format(key, subEvent[key])
                config['separatedPulseSeq'].append(subEvent)
    config['separatedPulseSeq'] = sorted(config['separatedPulseSeq'], key=lambda event: event['t'])


def emptyEvent(): 
    return {'w1': 0, 'Gx': 0, 'Gy': 0, 'Gz': 0, 'phase': 0, 'spoil': False}


def mergeEvent(event, event2merge, t):
    

    for channel in ['w1', 'Gx', 'Gy', 'Gz', 'phase']:
        if channel in event2merge:
            event[channel] += event2merge[channel]
    for text in ['RFtext', 'Gxtext', 'Gytext', 'Gztext', 'spoilText']:
        if text in event2merge:
            event[text] = event2merge[text]
    event['t'] = t
    return event


def detachEvent(event, event2detach, t):

    for channel in ['w1', 'Gx', 'Gy', 'Gz', 'phase']:
        if channel in event2detach:
            event[channel] -= event2detach[channel]
    for text in ['RFtext', 'Gxtext', 'Gytext', 'Gztext', 'spoilText']:
        if text in event and text in event2detach and event[text]==event2detach[text]:
            del event[text]
    event['t'] = t
    return event


def getPrescribedTimeVector(config, nTR):

    speedEvents = config['speed'] + [event for event in config['pulseSeq'] if any(['FA' in event, 'B1' in event])]
    speedEvents = sorted(speedEvents, key=lambda event: event['t'])
    
    kernelTime = np.array([])
    t = 0
    dt = 1e3 / config['fps'] * config['speed'][0]['speed'] 
    for event in speedEvents:
        kernelTime = np.concatenate((kernelTime, np.arange(t, event['t'], dt)), axis=None)
        t = max(t, event['t'])
        if 'speed' in event:
            dt = 1e3 / config['fps'] * event['speed'] 
        if 'FA' in event or 'B1' in event:
            RFdt = min(dt, 1e3 / config['fps'] * config['maxRFspeed']) 
            kernelTime = np.concatenate((kernelTime, np.arange(event['t'], event['t'] + event['dur'], RFdt)), axis=None)
            t = event['t'] + event['dur']
    kernelTime = np.concatenate((kernelTime, np.arange(t, config['TR'], dt)), axis=None)

    timeVec = np.array([])
    for rep in range(nTR): 
        timeVec = np.concatenate((timeVec, kernelTime + rep * config['TR']), axis=None)
    return np.unique(roundEventTime(timeVec))


def setupPulseSeq(config):
    checkPulseSeq(config)
    config['events'] = []
    ongoingEvents = []
    newEvent = emptyEvent() 
    newEvent['t'] = 0
    for i, event in enumerate(config['separatedPulseSeq']):
        eventTime = roundEventTime(event['t'])
        if eventTime==newEvent['t']:
            newEvent = mergeEvent(newEvent, event, eventTime)
        else:
            config['events'].append(dict(newEvent))
            newEvent = mergeEvent(newEvent, event, eventTime)
        ongoingEvents.append(event)
        sorted(ongoingEvents, key=lambda event: event['t'] + event['dur'], reverse=False)
        if event is config['separatedPulseSeq'][-1]:
            nextEventTime = roundEventTime(config['TR'])
        else:
            nextEventTime = roundEventTime(config['separatedPulseSeq'][i+1]['t'])
        for stoppingEvent in [event for event in ongoingEvents[::-1] if roundEventTime(event['t'] + event['dur']) <= nextEventTime]:
            config['events'].append(dict(newEvent))
            newEvent = detachEvent(newEvent, stoppingEvent, roundEventTime(stoppingEvent['t'] + stoppingEvent['dur']))
            ongoingEvents.pop()
    config['events'].append(dict(newEvent))

    config['kernelClock'] = getPrescribedTimeVector(config, 1)
    config['kernelClock'] = addEventsToTimeVector(config['kernelClock'], config['events'])
    if config['kernelClock'][-1] == config['TR']:
        config['kernelClock'] = config['kernelClock'][:-1]
    config['nFramesPerTR'] = len(config['kernelClock'])
    config['t'] = np.array([])
    for rep in range(-config['nDummies'], config['nTR']):
        config['t'] = np.concatenate((config['t'], roundEventTime(config['kernelClock'] + rep * config['TR'])), axis=None)
    config['t'] = np.concatenate((config['t'], roundEventTime(config['nTR'] * config['TR'])), axis=None) 
    config['kernelClock'] = np.concatenate((config['kernelClock'], config['TR']), axis=None) 
    

def setlocation(slices, config, key='locations'):
    
    if not isinstance(slices[0], list):
        slices = [slices]
    if not isinstance(slices[0][0], list):
        slices = [slices]
    if key=='M0' and not isinstance(slices[0][0][0], list):
        slices = [slices]
    if 'nz' not in config:
        config['nz'] = len(slices)
    elif len(slices)!=config['nz']:
        raise Exception('Config "{}": number of slices do not match'.format(key))
    if 'ny' not in config:
        config['ny'] = len(slices[0])
    elif len(slices[0])!=config['ny']:
        raise Exception('Config "{}": number of rows do not match'.format(key))
    if 'nx' not in config:
        config['nx'] = len(slices[0][0])
    elif len(slices[0][0])!=config['nx']:
        raise Exception('Config "{}": number of elements do not match'.format(key))
    if key=='M0' and len(slices[0][0][0])!=3:
        raise Exception('Config "{}": inner dimension must be of length 3'.format(key))
    return slices


def checkConfig(config):
    #检查和设定默认值
    if any([key not in config for key in ['TR', 'B0', 'speed', 'output']]):
        raise Exception('Config must contain "TR", "B0", "speed", and "output"')
    config['TR'] = roundEventTime(config['TR'])
    config['locSpacing'] = 0.001 #默认距离时间间隔
    config['fps'] = 15#默认帧时间间隔
    config['maxRFspeed'] = 0.001
    config['nTR'] = 1#
    if 'nDummies' not in config:
        config['nDummies'] = 0
    if 'nIsochromats' not in config:
        config['nIsochromats'] = 1
    if 'isochromatStep' not in config:
        if config['nIsochromats']>1:
            raise Exception('Please specify "isochromatStep" [ppm] in config')
        else:
            config['isochromatStep']=0
    if 'components' not in config:
        config['components'] = [{}]
    config['nComps'] = len(config['components'])#物质种类数量
    for c, comp in enumerate(config['components']):
        for (key, default) in [('name', ''), 
                               ('CS', 0), 
                               ('T1', np.inf), 
                               ('T2', np.inf), 
                               ('vx', 0), 
                               ('vy', 0), 
                               ('vz', 0), 
                               ('Dx', 0), 
                               ('Dy', 0), 
                               ('Dz', 0)]:
            if key not in comp:
                comp[key] = default
    config['w0'] = 2*np.pi*gyro*config['B0'] # Larmor振动频率[kRad/s]

    # 检查速度和位置
    if isinstance(config['speed'], Number):
        config['speed'] = [{'t': 0, 'speed': config['speed']}]
    
    config['speed'] = sorted(config['speed'], key=lambda event: event['t'])
    
    setupPulseSeq(config)
    
    config['locations'] = setlocation([[[1]]], config)
    
    for (FOV, n) in [('FOVx', 'nx'), ('FOVy', 'ny'), ('FOVz', 'nz')]:
        config[FOV] = config[n]*config['locSpacing'] 

    for output in config['output']:
        output['tRange'] = [0, config['nTR'] * config['TR']]
        output['dpi'] = 100
        output['freeze'] = []
        if output['type']=='3D':
            output['drawAxes'] = config['nx']*config['ny']*config['nz'] == 1
            output['azimuth'] = -78 
            output['elevation'] = None 

    config['background'] = {}
    config['background']['color'] = 'white'
    

def resampleOnPrescribedTimeFrames(vectors, config):
    config['tFrames'] = getPrescribedTimeVector(config, config['nTR'])
    newShape = list(vectors.shape)
    newShape[6] = len(config['tFrames'])
    resampledVectors = np.zeros(newShape)
    for x in range(newShape[0]):
        for y in range(newShape[1]):
            for z in range(newShape[2]):
                for c in range(newShape[3]):
                    for i in range(newShape[4]):
                        for dim in range(newShape[5]):
                            resampledVectors[x,y,z,c,i,dim,:] = np.interp(config['tFrames'], config['t'], vectors[x,y,z,c,i,dim,:])#线性插值


    for channel in ['RFalpha', 'Galpha', 'spoilAlpha']:
        alphaVector = np.zeros([len(config['tFrames'])])
        for i in range(len(alphaVector)):
            if i == len(alphaVector)-1:
                ks = np.where(config['t']>=config['tFrames'][i])[0]
            else:
                ks = np.where(np.logical_and(config['t']>=config['tFrames'][i], config['t']<config['tFrames'][i+1]))[0]
            alphaVector[i] = np.max(config[channel][ks])
        config[channel] = alphaVector


    for text in ['RFtext', 'Gtext']:
        textVector = np.full([len(config['tFrames'])], '', dtype=object)
        for i in range(len(textVector)):
            k = np.where(config['t']>=config['tFrames'][i])[0][0]
            textVector[i] = config[text][k]
        config[text] = textVector

    return resampledVectors


def fadeTextFlashes(config):
    
    decay = 1.0/(config['fps']) #间隔为时间倒数
    for channel in ['RFalpha', 'Galpha', 'spoilAlpha']:
        for i in range(1, len(config[channel])):
            if config[channel][i]==0:
                config[channel][i] = max(0, config[channel][i-1]-decay)


def run(configFile, leapFactor=1):
    with open(configFile, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise Exception('Error reading config file') from exc
    checkConfig(config)
    vectors = np.empty((config['nx'],config['ny'],config['nz'],config['nComps'],config['nIsochromats'],6,len(config['t'])))
    for z in range(config['nz']):
        for y in range(config['ny']):
            for x in range(config['nx']): #三维的位置
                for c, component in enumerate(config['components']):#通过enumerate函数方便分类
                    
                    if isinstance(config['locations'], list):
                        Meq = config['locations'][z][y][x]
                    else:
                        Meq = 0.0
                    
                    M0 = None
                    pos = [(x+.5-config['nx']/2)*config['locSpacing'],#方便显示
                           (y+.5-config['ny']/2)*config['locSpacing'],
                           (z+.5-config['nz']/2)*config['locSpacing']]
                    vectors[x,y,z,c,:,:,:] = simulateComponent(config, component, Meq, M0, pos)

    getText(config) 
    vectors = resampleOnPrescribedTimeFrames(vectors, config)
    fadeTextFlashes(config)
    delay = int(100/config['fps']*leapFactor)  
    outdir = './out'
    for output in config['output']:
        if output['file']:
            signal = np.sum(vectors[:,:,:,:,:,:3,:], (0,1,2,4))
            ffmpegWriter = FFMPEGwriter.FFMPEGwriter(config['fps'])
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, output['file'])
            
            output['freezeFrames'] = []
            for t in output['freeze']:
                output['freezeFrames'].append(np.argmin(np.abs(config['tFrames'] - t)))
            for frame in range(0, len(config['tFrames']), leapFactor):
                
                fig = plotFrame3D(config, vectors, frame, output,signal)
                plt.draw()

                filesToSave = []
                if frame in output['freezeFrames']:
                    filesToSave.append('{}_{}.png'.format('.'.join(outfile.split('.')[:-1]), str(frame).zfill(4)))

                ffmpegWriter.addFrame(fig)
                
                for file in filesToSave:
                    print('Saving frame {}/{} as "{}"'.format(frame+1, len(config['tFrames']), file))
                    plt.savefig(file, facecolor=plt.gcf().get_facecolor())
                plt.close()
            ffmpegWriter.write(outfile)

#通过yml文件读取，方便多种选择，参考了老师的代码
def parseAndRun():
    parser = argparse.ArgumentParser(description='Simulate magnetization vectors using Bloch equations and create animated gif')
    parser.add_argument('--configFile', '-c',
                        help="Name of configuration text file",
                        type=str,
                        default='')
    parser.add_argument('--leapFactor', '-l',
                        help="Leap factor for smaller filesize and fewer frames per second",
                        type=int,
                        default=1)
    args = parser.parse_args()
    run(args.configFile, args.leapFactor)

if __name__ == '__main__':
    run(r'SE.yml')


