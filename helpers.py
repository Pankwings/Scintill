import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Number of steps the channels in the waveform are separated by.
CHANNEL_SEPARATION_STEPS = 4
# Size of the columns.
DATA_COLUMN_SIZE = 512
# Index of the array that contains the actual data.
DATA_INDEX = 6

def load_waveformdata(dir_location, voltage):
    """
    Load pickled numpy array data from a given directory.
    dir_location is absolute path of the data directory.
    voltage is biasing voltage, only the files with given biasing voltage are loaded.
    """
    # Get the directory name from the given path.
    dir_name = os.path.dirname(dir_location)
    file_names = os.listdir(dir_location)
    
    # Create a zero ndarray with the required shape.
    wd = np.zeros((0, DATA_COLUMN_SIZE))
    num_file_read = 0
    for filename in file_names:
        # Construct absolute file paths.
        filename = os.path.join(dir_name, filename)
        if filename.endswith('.npy') and voltage in filename:
            d = np.array(np.load(filename, allow_pickle=True))[6]
            # Append the array only when the read array is of the the compatible
            # size.      
            if d.shape[1] == DATA_COLUMN_SIZE:
                wd = np.concatenate((wd, d))
                num_file_read+=1
            else:
                print('WARNING: Data column size mismatch for {}'.format(filename))
    #print(pd.DataFrame(wd))            
    if num_file_read == 0:
        print("No files with voltage = {}".format(voltage))
        
    return wd, num_file_read

# waveforms=waveformData[], sign= positive or negative signal err= error condidered start1=
def filter_channel_data(waveforms, start):
    """
    Extract specific channel signals from the given waveforms.
    and also adjust the waveform in positive signal response.
    """
    channel=[]           
    for row in range(start, len(waveforms), CHANNEL_SEPARATION_STEPS):
        # signal at that 512 trigger. So, it can be change according to our need.)

        
        if((abs(max(waveforms[row])))<(abs(min(waveforms[row])))):
            sign=-1
        else:
            sign=1
        # Old version code: thrsld=sign*err*min((waveforms[row]))
        
                # Iterate through the waveforms and filter signals above the threshold.
        # Set 0 for signals below the threshold.
        for col in waveforms[row]:
            # Waveform was negative to make it positive.
            # NOTE: Add more details.
            col = sign*col
            channel.append(col)
 
    #print(pd.DataFrame(channel))
    return channel

###http://databookuw.com/ Chapter 2-Denoising
def denoising(channel):
    signals = len(channel)
    fhat = np.fft.fft(channel,signals)
    PSD = fhat*np.conj(fhat)/signals
    ## Use the PSD to filter out noise
    indices = PSD > 0       # Find all freqs with large power
    fhat = indices * fhat     # Zero out small Fourier coeffs. in Y
    ffilt = np.fft.ifft(fhat) # Inverse FFT for filtered time signal
    return np.real(ffilt)

##------------------------------------------------------------------------------------------------
def fn_hist(channel):
    hchannel=[]
    period=512
    start=0
    for _ in range(0,len(channel),period):        # program for histrogram each channel
        maxx=max(channel[start:start+period])
        hchannel.append(maxx)
        start=start+period
    return hchannel

#------------------------------------------------------------------------------------------------
#"Averaging of signal of individual channel(average of all signal in channel)"

#channel= channel array variable,clr= colour of fig,chnm= channel number
def fn_comb( channel, chnm, no_file_read):
    avg_channel_one_set = {}
    v1 = 512    
    comb = [0]*v1    
    count = 0
    count0 = 0
    for num in range(no_file_read):
        comb=[0]*v1
        for i in range(int(len(channel)/no_file_read)):#int(len(channel)/no_file_read) = 512000
            if(count == v1):
                count=0
                count0+=v1
            comb[count] = comb[count] + channel[count+count0]
            count+=1
        for i in range(512):
            #averaging the signal of the channel(1000 trigger in a channel)
            comb[i]=comb[i]/1000 
        
        #Reading the content from the other .npy file i.e jumping within the same 1D variable after 512000 index. 
        avg_channel_one_set[num] = comb
    print_avg_signal( avg_channel_one_set, chnm, no_file_read)
    return avg_channel_one_set
    
    '''
    # OLD Code:
    count = 0
    v1 = 512
    comb1 = [0]*v1    #v1=512
    for num in range(no_file_read):
        for i in channel:
            if(count==v1):
                count=0
            comb1[count]=comb1[count]+i
            count+=1
        for i in range(512):
            comb1[i]=comb1[i]/1000 #averaging the signal of the channel(1000 trigger in a channel)
    '''    
def print_avg_signal( avg_channel_one_set, chnm, no_file_read):
    plt.title('Average of signal of channel %i (Averaging of all signal of the channel)' %chnm) 
    plt.figure(1)
    for i in range(no_file_read):
        plt.plot(avg_channel_one_set[i], label='Biasing Voltage %s' %(6+(0.1*i)))
        #plt.savefig('output%i.png'%chnm, dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()


