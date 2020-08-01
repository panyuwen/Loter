#####################################################################
## Note: VCF header required
## 		 ##fileformat=VCFv4.1
## 		 ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##		 ...
## Note: It is not currently possible to account for phasing errors in Loter local ancestry inference procedure 
##       when there are more than two ancestral populations. 
##       Haplotypic length of ancestry tracts will be affected and should not be used when there are more than 2 ancestral populations. 
##       Other statistics at the scale of genotypes are accurately inferred.
##       (mean ancestry in the population, mean ancestry per individual)
#####################################################################
## Uasge: /home/panyuwen/anaconda2/bin/python Loter.py -h
## Example: /home/panyuwen/anaconda2/bin/python Loter.py \
##				--target target.phase.vcf.gz \
##				--ref /phase/ref/phase.vcf.gz/file/path/list \
##				[--threads 10] [--correction T] [--genoanc F] [--out ./out]
## Output: a matrix of [number of SNPs * number of haplotypes]
## 			0 stands for the first ancestry population in the ref list
##			1 stands for the second ancestry population in the ref list, if there are only two ancestry involved
#####################################################################

import allel
import numpy as np
import pandas
import os
import sys
import time
import argparse
import loter.locanc.local_ancestry as lc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#####################################################################
##########                      Arguments                  ##########
#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, required=True, \
					help="phase.vcf.gz file, header required.")
parser.add_argument("--ref", type=str, required=True, \
					help="list of ref.phase.vcf.gz files")
parser.add_argument("--threads", type=int, required=False, default=10, \
					help="number the threads.")
parser.add_argument("--correction", type=str, required=False, default='T', choices=['T','F'], \
					help="whether to do local ancestry inference with phase correction.")
parser.add_argument("--genoanc", type=str, required=False, default='F', choices=['T','F'], \
					help="whether to estimate genotypic ancestry, only in the case of diploid organisms")
parser.add_argument("--out", type=str, required=False, default='./out', \
					help="prefix of output file name")
args = parser.parse_args()

with open(args.out+'.logfile','w') as f:
	f.write(time.strftime("%Y-%m-%d %X",time.localtime())+'\n')
	f.write(os.getcwd()+'\n\n')
	f.write('python ' + sys.argv[0] + '\n')
	f.write('	--target ' + args.target + '\n')
	f.write('	--ref ' + args.ref + '\n')
	f.write('	--threads ' + str(args.threads) + '\n')
	f.write('	--genoanc ' + args.genoanc + '\n')
	f.write('	--out ' + args.out + '\n')

#####################################################################
##########                        Main                     ##########
#####################################################################

## vcf2npy convert a VCF File to a numpy matrix with values 0, 1 and 2.
def vcf2npy(vcfpath):
    callset = allel.read_vcf(vcfpath)
    haplotypes_1 = callset['calldata/GT'][:,:,0]
    haplotypes_2 = callset['calldata/GT'][:,:,1]
    
    m, n = haplotypes_1.shape
    mat_haplo = np.empty((2*n, m))
    mat_haplo[::2] = haplotypes_1.T
    mat_haplo[1::2] = haplotypes_2.T
    
    return mat_haplo.astype(np.uint8)

target_data = vcf2npy(os.path.join('./', args.target))
reflist = pandas.read_csv(args.ref,header=None)
reflist = list(reflist[0])
ref_data = [vcf2npy(os.path.join('./', ref_file)) for ref_file in reflist]

## local ancestry inference with phase correction
## l_H: a list of "ancestral" or reference haplotypes matrices. Its length is equal to the number of ancestral populations.
## h_adm: a matrix of admixed haplotypes
## num_threads: number of threads for parallel computations
## range_lambda: list or 1-d array of candidate values (>0) for $\lambda$
## threshold: smoothing parameter in [0,1] for the phase correction module (=1 corresponds to no smoothing)
## rate_vote: bagging vote parameter in [0,1]
## nb_bagging: number of resampling in the bagging (positive interger)
plt.clf()
if (len(reflist) == 2) & (args.correction == 'T'):
	with open(args.out+'.logfile','a') as f:
		f.write("	--correction T\n")
	res_loter = lc.loter_smooth(l_H=ref_data, h_adm=target_data, num_threads=args.threads, \
				range_lambda=np.arange(1.5, 5.5, 0.5), threshold=0.90, rate_vote=0.5, nb_bagging=20)
	#np.savetxt(args.out, res_loter, fmt="%i")
	pandas.DataFrame(res_loter).T.to_csv(args.out+'.txt.gz',sep=' ',header=None,index=None,compression='gzip')
	## ancestry chunks visualization
	## Chunks of 0 correspond to the first element of the list, and chunks of 1 correspond to the second element of the ref list
	plt.imshow(res_loter, interpolation='nearest', aspect='auto')
	plt.colorbar()
	plt.savefig(args.out+'.png',format='png')
	plt.clf()
else:
	with open(args.out+'.logfile','a') as f:
		f.write("	--correction F\n")
	res_loter = lc.loter_local_ancestry(l_H=ref_data, h_adm=target_data, num_threads=args.threads, \
				range_lambda=np.arange(1.5, 5.5, 0.5), rate_vote=0.5, nb_bagging=20)
	#np.savetxt(args.out, res_no_impute[0], fmt="%i")
	pandas.DataFrame(res_loter[0]).T.to_csv(args.out+'.txt.gz',sep=' ',header=None,index=None,compression='gzip')
	## ancestry chunks visualization
	## Chunks of 0 correspond to the first element of the ref list
	plt.imshow(res_loter[0], interpolation='nearest', aspect='auto')
	plt.colorbar()
	plt.savefig(args.out+'.png',format='png')
	plt.clf()

## res_no_impute[0] contains the ancestry
## res_no_impute[1] contains the number of time that ancestry was picked in the bagging procedure.
## res_impute contains the genotypic ancestry. 
##            The genotypic ancestry corresponds to the paired haplotypic ancestries without order. 
##            For instance, if there are 3 ancestral populations, there are 6 possible ancestry values for genotypic ancestry.
if args.genoanc == 'T':
	res_impute, res_no_impute = lc.loter_local_ancestry(l_H=ref_data, h_adm=target_data, num_threads=args.threads, \
								range_lambda=np.arange(1.5, 5.5, 0.5), rate_vote=0.5, nb_bagging=20, default=False)
	pandas.DataFrame(res_impute).T.to_csv(args.out+'.Genotypic_Anc.txt.gz',sep=' ',header=None,index=None,compression='gzip')
	plt.imshow(res_impute, interpolation='nearest', aspect='auto')
	plt.colorbar()
	plt.savefig(args.out+'.Genotypic_Anc.png',format='png')
	plt.clf()
else:
	pass

with open(args.out+'.logfile','a') as f:
	f.write('\nFinnished.\n')
	f.write(time.strftime("%Y-%m-%d %X",time.localtime())+'\n')

print('Done Analysis.')
print('Have a Nice Day!')
