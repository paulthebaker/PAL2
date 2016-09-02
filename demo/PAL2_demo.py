#!/usr/bin/env python
from __future__ import division

import glob, os, sys
print sys.executable

# use non-interactive backend
from matplotlib import use
use('PDF')

import numpy as np
import PAL2
from PAL2 import PALmodels
from PAL2 import PALutils
from PAL2 import PALdatafile
from PAL2 import PALInferencePTMCMC as ptmcmc
from PAL2 import bayesutils as bu
import matplotlib.pyplot as plt

datadir = '/Users/ptb/Projects/nanograv/dat/NANOGrav_9y'
parFiles = glob.glob(datadir + 'par/*gls.par') 
timFiles = glob.glob(datadir + 'tim/*.tim')

# sort 
parFiles.sort()
timFiles.sort()

# make hdf5 file
h5filename = './h5file.hdf5'
df = PALdatafile.DataFile(h5filename)

# add pulsars to file
for p, t in zip(parFiles, timFiles):
    df.addTempoPulsar(p, t, iterations=0)


# just use one pulsar for now
pulsars = ['1643-1224']
model = PALmodels.PTAmodels(h5filename, pulsars=pulsars)

# initialize a model that will include a search for 
# powerlaw red noise + efac (efac is set by default) + equad + ecorr
fullmodel = model.makeModelDict(incRedNoise=True, noiseModel='powerlaw', 
                                nfreqs=50, incJitterEquad=True, incEquad=True,
                                likfunc='mark6')

# initialize model
model.initModel(fullmodel, memsave=True, fromFile=False, verbose=True) # trying verbose=True to see output on cluster

# get names of varying parameters
pars = model.get_varying_parameters()
print(pars)


#####
##  SETUP MCMC FOR SINGLE PULSAR NOISE ANALYSIS
#####

# set initial parameters. This will randomly draw parameters from the prior.
p0 = model.initParameters()

# Set initial covariance jump sizes. 
# This uses the pwidth key in the parameter dictionaries
cov = model.initJumpCovariance()

# define likelihood and prior function
lnlike = model.mark6LogLikelihood
lnprior = model.mark3LogPrior


# parameter groups allow you to only jump in certain groups of parameters at a time
# for example you can do the following if you want to separate out Red noise jumps
ind = []

# pull out powerlaw parameters from overall parameter array
ids = model.get_parameter_indices('powerlaw', corr='single', split=True)

# add to list
[ind.append(id) for id in ids]

# make list of lists that has all parameters and then individual indices defined above
ind.insert(0, range(len(p0)))

# can give this list to the sampler by including the keyword argument groups=ind

# setup sampler
# make keyword dictionary for jitter parameter
loglkwargs = {'incJitter': True}

sampler = ptmcmc.PTSampler(len(p0), lnlike, lnprior, cov,  outDir='./chains/',
                           loglkwargs=loglkwargs)

# can add additional custom jump proposals with a given weight [2=small, 5=medium, 10=Large, 20=very large]
# the weights give some indication as to how often a given proposal is used in the cycle
#sampler.addProposalToCycle(name_of_function, weight)

# run sampler for a max of 100 000 samples or for 1000 effective samples
N = 100000
Neff = 1000
sampler.sample(p0, N, neff=Neff, writeHotChains=True)

print "I'm chain number {rank:d}/{N:d}".format(rank=1+sampler.MPIrank, N=sampler.nchain)

#####
##  POST PROCESSING
#####

# generate plots directory
if not os.path.exists('./plots'):
    try:
        os.makedirs('./plots')
    except OSError:
        pass

# read in chain file and set burn in to be 25% of chain length
chain = np.loadtxt('chains/chain_1.txt')
burn = 0.25 * chain.shape[0]

# plot posterior values to check for convergence
fig1 = plt.figure(1)
ax = fig1.add_subplot(1,1,1)
ax.plot(chain[burn:,-4])
ax.set_ylabel('log-posterior')
fig1.savefig("plots/chain.pdf")

# plot acceptance rate for different jumps
plt.rcParams['font.size'] = 10
jumpfiles = glob.glob('chains/*jump.txt')
jumps = map(np.loadtxt, jumpfiles)
fig2 = plt.figure(2)
ax = fig2.add_subplot(1,1,1)
for ct, j in enumerate(jumps):
    ax.plot(j, label=jumpfiles[ct].split('/')[-1].split('_jump.txt')[0])
ax.legend(loc='best', frameon=False)
ax.set_ylabel('Acceptance Rate')
fig2.savefig("plots/accept.pdf")

# plot triangle plot of fit parameters
plt.rcParams['font.size'] = 6
ax = bu.triplot(chain[burn:,:-4], labels=pars[:], tex=False, figsize=(20,15))
plt.savefig("plots/param_tri.pdf")
