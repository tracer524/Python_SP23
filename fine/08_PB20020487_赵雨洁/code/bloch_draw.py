
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





colors = {  'bg':       [1,1,1], 
            'circle':   [0,0,0,.03],
            'axis':     [.5,.5,.5],
            'text':     [.05,.05,.05], 
            'spoilText':[.5,0,0],
            'RFtext':   [0,.5,0],
            'Gtext':    [80/256,80/256,0],
            'comps': [  [.3,.5,.2],
                        [.1,.4,.5],
                        [.5,.3,.2],
                        [.5,.4,.1],
                        [.4,.1,.5],
                        [.6,.1,.3]],
            'boards': { 'w1': [.5,0,0],
                        'Gx': [0,.5,0],
                        'Gy': [0,.5,0],
                        'Gz': [0,.5,0]
                        },
            'kSpacePos': [1, .5, 0]
            }

def getText(config):
    ''' Get opacity and display text pulseSeq event text flashes in 3D plot and store in config.
    
    Args:
        config: configuration dictionary.
        
    '''

    # Setup display text related to pulseSeq events:
    config['RFtext'] = np.full([len(config['t'])], '', dtype=object)
    config['Gtext'] = np.full([len(config['t'])], '', dtype=object)
    config['spoiltext'] = 'spoiler'
    config['RFalpha'] = np.zeros([len(config['t'])])   
    config['Galpha'] = np.zeros([len(config['t'])])
    config['spoilAlpha'] = np.zeros([len(config['t'])])   
        
    for rep in range(config['nTR']):
        TRstartFrame = rep * config['nFramesPerTR']
        for i, event in enumerate(config['events']):
            firstFrame, lastFrame = getEventFrames(config, i)        
            firstFrame += TRstartFrame
            lastFrame += TRstartFrame

            if 'RFtext' in event:
                config['RFtext'][firstFrame:] = event['RFtext']
                config['RFalpha'][firstFrame:lastFrame+1] = 1.0
            if any('{}text'.format(g) in event for g in ['Gx', 'Gy', 'Gz']): # gradient event
                Gtext = ''
                for g in ['Gx', 'Gy', 'Gz']:
                    if '{}text'.format(g) in event:
                        Gtext += '  ' + event['{}text'.format(g)]
                config['Gtext'][firstFrame:] = Gtext
                config['Galpha'][firstFrame:lastFrame+1] = 1.0
            if 'spoil' in event and event['spoil']:
                config['spoilAlpha'][firstFrame] = 1.0
def getEventFrames(config, i):
    '''Get first and last frame of event i in config['events'] in terms of config['t']

    Args:
        config:         configuration dictionary.
        i:              event index
        
    Returns:
        firstFrame:     index of first frame in terms of config['t']
        lastFrame:      index of last frame in terms of config['t']
        
    '''
    try:
        firstFrame = np.where(config['t']==config['events'][i]['t'])[0][0]
    except IndexError:
        print('Event time not found in time vector')
        raise
    
    if i < len(config['events'])-1:
        nextEventTime = config['events'][i+1]['t']
    else:
        nextEventTime = config['TR']
    try:
        lastFrame = np.where(config['t']==nextEventTime)[0][0]
    except IndexError:
        print('Event time not found in time vector')
        raise
    return firstFrame, lastFrame

class Arrow3D(FancyArrowPatch):
    '''Matplotlib FancyArrowPatch for 3D rendering.'''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        self.do_3d_projection()
        FancyArrowPatch.draw(self, renderer)
    
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def plotFrame3D(config, vectors, frame, output,signal):
    '''Creates a plot of magnetization vectors in a 3D view.
    
    Args:
        config: configuration dictionary.
	vectors:    numpy array of size [nx, ny, nz, nComps, nIsochromats, 3, nFrames].
        frame:  which frame to plot.
        output: specification of desired output (dictionary from config).

    Returns:
        plot figure.

    '''
    nx, ny, nz, nComps, nIsoc = vectors.shape[:5]

    # Create 3D axes
    
    aspect = 1
    figSize = 5 # figure size in inches
    canvasWidth = figSize*2
    canvasHeight = figSize*aspect
    fig = plt.figure(figsize=(canvasWidth, canvasHeight), dpi=output['dpi'])
    axLimit = max(nx,ny,nz)/2+.5
    #plot1
  
    ax = plt.axes(projection='3d', xlim=(-axLimit,axLimit), ylim=(-axLimit,axLimit), zlim=(-axLimit,axLimit), fc=colors['bg'])
    
    if nx*ny*nz>1 and not config['collapseLocations']:
        ax.view_init(azim=output['azimuth'], elev=output['elevation'])
    ax.set_axis_off()
    width = 1.6 # to get tight cropping
    height = width/aspect
    left = (1-width)/2
    bottom = (1-height)/2
    left -= .02
    bottom += -.075
   
    ax.set_position([left, bottom, width, height])

    
    for i in ["x", "y", "z"]:
            circle = Circle((0, 0), 1, fill=True, lw=1, fc=colors['circle'])
            ax.add_patch(circle)
            art3d.pathpatch_2d_to_3d(circle, z=0, zdir=i)

        # Draw x, y, and z axes
    ax.plot([-1, 1], [0, 0], [0, 0], c=colors['axis'], zorder=-1)  # x-axis
    ax.text(1.08, 0, 0, r'$x^\prime$', horizontalalignment='center', color=colors['text'])
    ax.plot([0, 0], [-1, 1], [0, 0], c=colors['axis'], zorder=-1)  # y-axis
    ax.text(0, 1.12, 0, r'$y^\prime$', horizontalalignment='center', color=colors['text'])
    ax.plot([0, 0], [0, 0], [-1, 1], c=colors['axis'], zorder=-1)  # z-axis
    ax.text(0, 0, 1.05, r'$z$', horizontalalignment='center', color=colors['text'])

    # Draw title:
    fig.text(.5, 1, config['title'], fontsize=14, horizontalalignment='center', verticalalignment='top', color=colors['text'])

    # Draw time
    time = config['tFrames'][frame%(len(config['t'])-1)] # frame time [msec]
    time_text = fig.text(0, 0, 'time = %.1f msec' % (time), color=colors['text'], verticalalignment='bottom')

    # TODO: put isochromats in this order from start
    order = [int((nIsoc-1)/2-abs(m-(nIsoc-1)/2)) for m in range(nIsoc)]
    arrowheadThres = 0.075 * axLimit # threshold on vector magnitude for arrowhead shrinking
    projection = np.array([np.cos(np.deg2rad(ax.azim)) * np.cos(np.deg2rad(ax.elev)),
                           np.sin(np.deg2rad(ax.azim)) * np.cos(np.deg2rad(ax.elev)),
                           np.sin(np.deg2rad(ax.elev))])
    
    pos = [0,0,0]

    # Draw magnetization vectors
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for c in range(nComps):
                    for m in range(nIsoc):
                        col = colors['comps'][(c) % len(colors['comps'])]
                        M = vectors[x,y,z,c,m,:3,frame]
                        
                        pos = vectors[x,y,z,c,m,3:,frame]/config['locSpacing']
                                             
                        Mnorm = np.sqrt((np.linalg.norm(M)**2 - np.dot(M, projection)**2)) # vector norm in camera projection
                        if Mnorm > arrowheadThres:
                            arrowScale = 20
                        else:
                            arrowScale = 20*Mnorm/arrowheadThres # Shrink arrowhead for short arrows
                        alpha = 1.-2*np.abs((m+.5)/nIsoc-.5)
                        ax.add_artist(Arrow3D(  [pos[0], pos[0]+M[0]], 
                                                [-pos[1], -pos[1]+M[1]],
                                                [-pos[2], -pos[2]+M[2]], 
                                                mutation_scale=arrowScale,
                                                arrowstyle='-|>', shrinkA=0, shrinkB=0, lw=2,
                                                color=col, alpha=alpha, 
                                                zorder=order[m]+nIsoc*int(100*(1-Mnorm))))

    # Draw "spoiler" and "FA-pulse" text
    fig.text(1, .94, config['RFtext'][frame], fontsize=14, alpha=config['RFalpha'][frame],
            color=colors['RFtext'], horizontalalignment='right', verticalalignment='top')
    fig.text(1, .88, config['Gtext'][frame], fontsize=14, alpha=config['Galpha'][frame],
            color=colors['Gtext'], horizontalalignment='right', verticalalignment='top')
    fig.text(1, .82, config['spoiltext'], fontsize=14, alpha=config['spoilAlpha'][frame],
            color=colors['spoilText'], horizontalalignment='right', verticalalignment='top')

    # Draw legend:
    for c in range(nComps):
        col = colors['comps'][(c) % len(colors['comps'])]
        ax.plot([0, 0], [0, 0], [0, 0], '-', lw=2, color=col, alpha=1.,
                    label=config['components'][c]['name'])
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend([plt.Line2D((0, 1), (0, 0), lw=2, color=colors['comps'][(c) %
                                len(colors['comps'])]) for c, handle in enumerate(
                                handles)], labels, loc=2, bbox_to_anchor=[
                                -.025, .94])
    leg.draw_frame(False)
    for text in leg.get_texts():
        text.set_color(colors['text'])
    
    xmin, xmax = output['tRange']
    
    ymin, ymax = 0, 0.5
 
    ax1 = fig.add_axes([0.8,0.05,0.2,0.2])
   
    ax1.grid()
   
    plt.xlabel('t', horizontalalignment='right', color=colors['text'])
    ax1.xaxis.set_label_coords(1.1, .075)
    plt.ylabel('$B$', rotation=0, color=colors['text'])
        
   

    # draw x and y axes as arrows
    bbox = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height  # get width and height of axes object
    hw = 1/25*(ymax-ymin)  # manual arrowhead width and length
    hl = 1/25*(xmax-xmin)
    yhw = hw/(ymax-ymin)*(xmax-xmin) * height/width  # compute matching arrowhead length and width
    yhl = hl/(xmax-xmin)*(ymax-ymin) * width/height
    ax1.set_xlabel('t/ms')
    ax1.set_ylabel('B')
    # Draw magnetization vectors
    nComps = signal.shape[0]
    ax1.plot(config['tFrames'][:frame+1], 0.5*(np.linalg.norm(signal[c,:2,:frame+1], axis=0)), '-', lw=2, color=col)
        # plot sum component if both water and fat (special case)

    return fig




