import pycbc
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.filter import match, sigmasq
from pycbc import conversions, psd, distributions
import time, h5py, sys, numpy as np
from tqdm import tqdm
import uuid, argparse
import pandas as pd

#parser = argparse.ArgumentParser(description=__doc__)
#parser.add_argument('--num-samples', required=True,
#                    help='The number of injection samples to fit against template bank.')

# Read in the bank
four = h5py.File('bank.hdf','r')

# read in PSD
psdtxt = np.loadtxt('o3psd.txt')

length = len(psdtxt[:,0])
delta_f = psdtxt[1,0] - psdtxt[0,0]
flow = 15
psd = pycbc.psd.read.from_txt('o3psd.txt',
                             length,
                             delta_f,
                             flow,
                             is_asd_file=False)
# generate random parameters 
lowmc = conversions.mchirp_from_mass1_mass2(1,1)
highmc = conversions.mchirp_from_mass1_mass2(2,2)
dist = distributions.Uniform(injmc=(lowmc,highmc), injspin=(-0.95,0.95))

def save_output(injpar,output):
    '''
    save the output as a csv file
    
    Parameters
    -----------
    injpar: dict
        injection parameters
    outpar: dict
        max_match parameters
    '''
    df1 = pd.DataFrame.from_dict(injpar)
    df2 = pd.DataFrame.from_dict(output)
    df3 = df1.join(df2)
    df3.to_csv('result/match-'+str(uuid.uuid4())[:8]+'.csv')
    
# calculate fitting factors
def fitting_factor(inj_sample, base_bank=four, psd=psd,
                   flow=45, fhigh=1024,df=0.125,tau_tol=2):
    '''
    Parameters:
    -----------
    inj_sample: Injections, should contain mass and spin
    base_bank: Template bank to be verified
    psd: PSD curve
    
    '''
    print('start')
    base_bank_tau0 = conversions.tau0_from_mass1_mass2(
                                            mass1=base_bank['mass1'][:],
                                            mass2=base_bank['mass2'][:],
                                            f_lower=20)
    out_par = {} # returned object
    for k in range(len(inj_sample)):#loop over all injection samples
	# injection with equal mass sources
        injmass = conversions.mass1_from_mchirp_q(inj_sample['injmc'][k],1)
        hpinj, _ = get_fd_waveform(approximant="TaylorF2",
                           mass1=injmass,
                           mass2=injmass,
                           spin1z=inj_sample['injspin'][k],
                           spin2z=inj_sample['injspin'][k],
                           delta_f=df,
                           f_lower=20,#for the injection (a reprentation of real signals, we start at 20 Hz)
                           f_final=fhigh)#injection waveform

        inj_tau0 = conversions.tau0_from_mass1_mass2(
                                           mass1=injmass,
                                           mass2=injmass,
                                           f_lower=20)# injection waveform duration
        
        #select those in the bank close to the injection waveform
        index = np.where( np.abs(inj_tau0 - base_bank_tau0) < tau_tol )[0]

        max_match = 0 
        max_par = {} #initialization
        for i in index:
            #print(i)
            # extract the base bank parameters
            par = {}
            for k in base_bank:
                par[k] = base_bank[k][i]
            par.pop('template_duration')#approximant
            par.pop('f_lower')
            
            #template waveform
            hpbank, __ = get_fd_waveform(approximant = "TaylorF2",
                                     **par,
                                     delta_f=df,
                                     f_lower=45, #for the template bank, starting at 45 Hz
                                     f_final=fhigh)
            #calculate fitting factor
            cache_match, _ = match(hpinj,
                               hpbank,
                               psd = psd,
                               low_frequency_cutoff = 20, #calculate the match from 20 Hz
                               high_frequency_cutoff = fhigh)
            
            #save the maximized fitting factor
            if cache_match > max_match:
                max_match = cache_match
                max_par = par 
        
        #save everything in a single dict, merge max_match into max_par
        max_par['max_match'] = max_match 
        
        # initialization
        if len(out_par) == 0:
            for k in max_par.keys():
                out_par[k] = [] 
        # append results in a dict
        for k in max_par.keys():
            out_par[k].append(max_par[k]) 
    
    save_output(inj_sample, out_par)

samples = dist.rvs(int(sys.argv[1]))
fitting_factor(samples)
