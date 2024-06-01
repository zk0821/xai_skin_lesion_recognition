.PHONY: hello container ham10000 hpc_ham10000 hpc_normalize hpc_res_net hpc_dense_net hpc_efficient_net hpc_transformer hpc_ci_net

hello:
	echo "Hello World!"

container:
	chmod u+x containers/create_container.sh
	./containers/create_container.sh

ham10000:
	python src/data/ham10000.py

hpc_ham10000:
	chmod u+x src/scripts/ham10000.sh
	sbatch src/scripts/ham10000.sh

hpc_normalize:
	chmod u+x src/scripts/ham_normalize.sh
	sbatch src/scripts/ham_normalize.sh

hpc_res_net:
	sbatch src/scripts/res_net.sh

hpc_dense_net:
	sbatch src/scripts/dense_net.sh

hpc_efficient_net:
	sbatch src/scripts/efficient_net.sh

hpc_transformer:
	sbatch src/scripts/transformer.sh

hpc_ci_net:
	sbatch src/scripts/ci_net.sh