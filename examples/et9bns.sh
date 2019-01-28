# MCMC runs for each BNS system
PEDATA0="input/BNS_0.csv"
#PEDATA1="input/BNS_1.csv" This data file has a suspiciosly large range of q's
PEDATA2="input/BNS_2.csv"
PEDATA3="input/BNS_3.csv"
PEDATA4="input/BNS_4.csv"
PEDATA5="input/BNS_5.csv"
PEDATA6="input/BNS_6.csv"
PEDATA7="input/BNS_7.csv"
PEDATA8="input/BNS_8.csv"
PEDATA9="input/BNS_9.csv"

# Output directory
OUTDIR="et9bns_output"

# Parameters for constructing the bounded_2d_kde and gridding it
QKDEBOUNDMIN=0.1
LAMBDATKDEBOUNDMAX=4000
GRIDSIZE=500

# Prior bounds
QMIN=0.5
LAMBDATMAX=4000
MMIN=0.7
MMAX=3.2
MAXMASSMIN=1.93
MAXMASSMAX=3.2
CSMIN=1.1

# Sampling parameters
NWALKERS=128

# Very accurate run
NITER=10000
NBURNIN=8000
NTHIN=10
NSAMPLE=10000

####### Start eosinference ########

# Create an output directory
mkdir $OUTDIR

# Generate pseudolikelihood function ln(p)(q, lambdat) for each BNS event
python ../bin/generate_likelihood.py \
--pefiles $PEDATA0 $PEDATA2 $PEDATA3 $PEDATA4 $PEDATA5 $PEDATA6 $PEDATA7 $PEDATA8 $PEDATA9 \
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
--eoslabels "$\log(p_1)$" "$\Gamma_1$" "$\Gamma_2$" "$\Gamma_3$" \
--eosname piecewise_polytrope_gamma_params \
--eostruths 34.384 3.005 2.988 2.851
