#!/bin/bash
###########################
#
#Scritp that reads parameter file and
#runs the integration routine sequentially for each parameter
#in the file
#
#
###########################

#default parameters, one of these is systematically replaced with the values in the parameter_file
Lattice_size=31
mu=0
J=4


#needed prerequisites for the run
parameter_file='params_mu'
dire_to_mods='../Mods/'
pow=$PWD

#Reading parameter file
param_arr=$(awk -F= '{print $1}' ${parameter_file})
echo ${param_arr}

jobname="musweep_updt_${Lattice_size}_${J}"  #JOBNAME importan to declare -has to be descriptive

#General info about the job
date_in="`date "+%Y-%m-%d-%H-%M-%S"`"
echo "${date_in}" >inforun
echo '....Jobs started running' >>  inforun

#Temporary directories where the runs will take place
dire_to_temps="../temp/temp_${jobname}_${date_in}"
rm -rf "${dire_to_temps}"
mkdir "${dire_to_temps}"

#loop over the parameters
for param_val in ${param_arr[@]}; do

	#create one temporary directory per parameter
	dire=""${dire_to_temps}"/${jobname}_${param_val}"
	rm -rf "${dire}"
	mkdir -vp "${dire}"


    cp ${dire_to_mods}Hamiltonian.py  "${dire}"
    cp ${dire_to_mods}Lattice.py  "${dire}"
    cp ${parameter_file}  "${dire}"

	#entering the temp directory, running and coming back
	cd "${dire}"
	echo "parameters: lattice size" ${Lattice_size} " J " ${J}  " nu "  ${param_val} >> output.out

	nohup time python3 -u Hamiltonian.py  ${Lattice_size} ${param_val} ${J} >> output.out &
	
	cd "${pow}"
	sleep 1

done

wait

#general info about the job as it ends
date_fin="`date "+%Y-%m-%d-%H-%M-%S"`"
echo "${date_fin}" >>inforun
echo 'Jobs finished running'>>inforun

#moving files to the data directory and tidying up
dire_to_data="../data/${jobname}_${date_fin}"
mkdir "${dire_to_data}"
mv "${dire_to_temps}"/* "${dire_to_data}"
mv inforun "${dire_to_data}"
rm -r "${dire_to_temps}"
