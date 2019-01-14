# TODO: add vsmax to the list of things you calculate

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import numpy as np
import h5py

import equationofstate as eospp
import runemcee
import postprocess

parser = argparse.ArgumentParser(description="Calculate the pseudolikelihood for each BNS system.")
required = parser.add_argument_group('Required named arguments')
required.add_argument('--priorfile', required=True, help='hdf5 output file for prior emcee run.')
required.add_argument('--posteriorfile', required=True, help='hdf5 output file for posterior emcee run.')
required.add_argument('--outfile', required=True, help='hdf5 output file for postprocessed metadata.')
parser.add_argument('--nburnin', type=int, default=0, help='Skip the first n steps from emcee files.')
parser.add_argument('--nthin', type=int, default=10, help='Only extract every nth step from emcee files.')
parser.add_argument('--nsample', type=int, default=1000, help='Use this many samples when calculating metadata.')
args = parser.parse_args()
print('Arguments from command line: {}'.format(args))
print('Results will be saved in {}'.format(args.outfile))


print('Loading emcee runs for prior and posterior.')
mc_mean_prior, lnprob_prior, samples_prior = runemcee.load_emcee_samples(args.priorfile)
mc_mean_post, lnprob_post, samples_post = runemcee.load_emcee_samples(args.posteriorfile)

# Number of BNS and EOS parameters
nparams = samples_post.shape[2]
nbns = len(mc_mean_post)
dim_eos = nparams - nbns
print('The emcee run had {} parameters ({} BNS and {} EOS).'.format(nparams, nbns, dim_eos))

print('Downsampling and flattening the emcee runs.')
q_prior, eos_prior = postprocess.q_eos_samples_from_emcee_samples(
    samples_prior,
    nburnin=args.nburnin, nthin=args.nthin, nsample=args.nsample, dim_eos=dim_eos)
q_post, eos_post = postprocess.q_eos_samples_from_emcee_samples(
    samples_post,
    nburnin=args.nburnin, nthin=args.nthin, nsample=args.nsample, dim_eos=dim_eos)


# TODO: This should not be hardcoded
eos_class_reference = eospp.EOS4ParameterPiecewisePolytropeGammaParams


print('Calculating Mmax, R(M) curve, Lambda(M) curve for each EOS sample.')
ms = np.linspace(0.5, 3.5, 1000)
ns_properties_prior = postprocess.ns_properties_from_eos_samples(eos_prior, eos_class_reference, ms)
ns_properties_post = postprocess.ns_properties_from_eos_samples(eos_post, eos_class_reference, ms)


single_bns_properties_list = []
for i in range(nbns):
    print('Evaluating [q, lambdat, m1, m2, r1, r2, l1, l2] for each EOS sample for BNS {}.'.format(i))
    mc_mean = mc_mean_post[i]
    q_samples = q_post[:, i]
    eos_samples = eos_post
    # print(mc_mean)
    # print(q_samples.shape)
    # print(eos_samples.shape)
    single_bns_properties = postprocess.single_event_ns_properties_from_samples(
        mc_mean, q_samples, eos_samples, eos_class_reference)
    single_bns_properties_list.append(single_bns_properties)


print('Saving metadata.')
f = h5py.File(args.outfile)

# Save metadata from downsampled prior run
group = f.create_group('prior')
group['eos_samples'] = ns_properties_prior['eos']
group['mmax_samples'] = ns_properties_prior['mmax']
group['mass_grid'] = ns_properties_prior['mass']
group['radius_curves'] = ns_properties_prior['radius']
group['lambda_curves'] = ns_properties_prior['lambda']

# Save metadata from downsampled posterior run
group = f.create_group('posterior')
group['eos_samples'] = ns_properties_post['eos']
group['mmax_samples'] = ns_properties_post['mmax']
group['mass_grid'] = ns_properties_post['mass']
group['radius_curves'] = ns_properties_post['radius']
group['lambda_curves'] = ns_properties_post['lambda']

# Save metadata for each BNS system from downsampled posterior run
for i in range(nbns):
    groupname = 'bns_{}'.format(i)
    group = f.create_group(groupname)
    group.attrs['mc_mean'] = mc_mean_post[i]
    group['q'] = single_bns_properties_list[i]['q']
    group['lambdat'] = single_bns_properties_list[i]['lambdat']
    group['m1'] = single_bns_properties_list[i]['m1']
    group['m2'] = single_bns_properties_list[i]['m2']
    group['r1'] = single_bns_properties_list[i]['r1']
    group['r2'] = single_bns_properties_list[i]['r2']
    group['l1'] = single_bns_properties_list[i]['l1']
    group['l2'] = single_bns_properties_list[i]['l2']

f.close()
