# MCMC runs for each BNS system
PEDATA0="input/RR30_reweight.csv"

# Output directory
OUTDIR="gw170817bhremnant_output"

# Parameters for constructing the bounded_2d_kde and gridding it
QKDEBOUNDMIN=0.125
LAMBDATKDEBOUNDMAX=5000
GRIDSIZE=250

# Prior bounds
QMIN=0.125
LAMBDATMAX=5000
MMIN=0.5
MMAX=2.2
MAXMASSMIN=1.93
MAXMASSMAX=2.2
CSMIN=1.1

# Sampling parameters
NWALKERS=64

# Fast run
NITER=100
NBURNIN=20
NTHIN=5
NSAMPLE=1000

# Accurate run
# NITER=1000
# NBURNIN=200
# NTHIN=10
# NSAMPLE=5000

# Very accurate run
# NITER=10000
# NBURNIN=5000
# NTHIN=10
# NSAMPLE=10000

####### Start eosinference ########

# Create an output directory
mkdir $OUTDIR

# Generate pseudolikelihood function ln(p)(q, lambdat) for each BNS event
python ../bin/generate_likelihood.py \
--pefiles $PEDATA0 \
--outfile ${OUTDIR}/pseudolikelihood.hdf5 \
--qmin $QKDEBOUNDMIN --lambdatmax $LAMBDATKDEBOUNDMAX --gridsize $GRIDSIZE

# Make output page for likelihood function
python ../bin/generate_likelihood_plot_page.py \
--infile ${OUTDIR}/pseudolikelihood.hdf5 \
--outdir ${OUTDIR}

#########################################################################
# Sample the prior and posterior in parallel by running in background (&)
# Prior
python ../bin/sample_distribution.py \
--infile ${OUTDIR}/pseudolikelihood.hdf5 \
--outfile ${OUTDIR}/prior.hdf5 \
--eosname piecewise_polytrope_gamma_params \
--distribution prior \
--nwalkers $NWALKERS --niter $NITER --nthin 1 \
--qmin $QMIN --lambdatmax $LAMBDATMAX \
--mmin $MMIN --mmax $MMAX \
--maxmassmin $MAXMASSMIN --maxmassmax $MAXMASSMAX --csmax $CSMIN &

# Sample the posterior
python ../bin/sample_distribution.py \
--infile ${OUTDIR}/pseudolikelihood.hdf5 \
--outfile ${OUTDIR}/posterior.hdf5 \
--eosname piecewise_polytrope_gamma_params \
--distribution posterior \
--nwalkers $NWALKERS --niter $NITER --nthin 1 \
--qmin $QMIN --lambdatmax $LAMBDATMAX \
--mmin $MMIN --mmax $MMAX \
--maxmassmin $MAXMASSMIN --maxmassmax $MAXMASSMAX --csmax $CSMIN &

# Wait for both processes (prior and posterior) to finish before proceeding
wait
#########################################################################

# Calculate NS properties for the downsampled parameters (burnin removed, thinned)
python ../bin/calculate_ns_properties.py \
--priorfile ${OUTDIR}/prior.hdf5 \
--posteriorfile ${OUTDIR}/posterior.hdf5 \
--eosname piecewise_polytrope_gamma_params \
--outfile ${OUTDIR}/ns_properties.hdf5 \
--nburnin $NBURNIN --nthin $NTHIN --nsample $NSAMPLE

# Generate output html page
python ../bin/generate_eos_output_page.py \
--infile ${OUTDIR}/ns_properties.hdf5 \
--priorfile ${OUTDIR}/prior.hdf5 \
--posteriorfile ${OUTDIR}/posterior.hdf5 \
--outdir ${OUTDIR} \
--eoslabels "$\log(p_1)$" "$\Gamma_1$" "$\Gamma_2$" "$\Gamma_3$"
