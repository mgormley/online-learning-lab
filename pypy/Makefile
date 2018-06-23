DATA=../data
SIM=$(DATA)/simple
SIM_TAT=$(DATA)/simple/tat
SIM_DSL=$(DATA)/simple/dsl
FEAT=$(DATA)/features
FEAT_TAT=$(DATA)/features/tat
FEAT_DSL=$(DATA)/features/dsl
OUT=output
TEAM=UPDATE_TEAM_IN_MAKEFILE
PYTHON=pypy3

all:
	echo "Use one of the commands."
	
setup:
	pip install --user mmh3

	mkdir -p $(DATA)
	mkdir -p $(OUT)
	
	cd $(DATA) && wget http://www.cs.cmu.edu/~mgormley/temp/langid.tgz
	tar -xzf $(DATA)/langid.tgz -C $(DATA)
	
submit:
	echo "Run the following command:"
	echo "scp $(OUT)/test-{tat,dsl}.txt <username>@bridges.psc.edu:/pylon2/ci560op/mgormley/submissions/$(TEAM)/"

dsl: dslf dslc

dslf:
	mkdir -p $(FEAT_DSL)
	$(PYTHON) features.py -i $(SIM_DSL)/train.txt -o $(FEAT_DSL)/train.txt
	$(PYTHON) features.py -i $(SIM_DSL)/dev.txt -o $(FEAT_DSL)/dev.txt
	$(PYTHON) features.py -i $(SIM_DSL)/test-blind.txt -o $(FEAT_DSL)/test-blind.txt
	cp $(SIM_DSL)/labels.txt $(FEAT_DSL)/labels.txt

dslc:
	$(PYTHON) classify.py \
			--train $(FEAT_DSL)/train.txt \
			--dev $(FEAT_DSL)/dev.txt \
			--test $(FEAT_DSL)/test-blind.txt \
			--test_out $(OUT)/test-dsl.txt \
			--labels $(FEAT_DSL)/labels.txt 
			#--dev_iters 50000

dslcfast:
	$(PYTHON) classify.py \
			--train $(FEAT_DSL)/train.txt \
			--dev $(FEAT_DSL)/dev.txt \
			--test $(FEAT_DSL)/test-blind.txt \
			--test_out $(OUT)/test-dslfast.txt \
			--labels $(FEAT_DSL)/labels.txt \
			--dev_iters 200 \
			--train_max 1000 --dev_max 100 --test_max 100 
	
tat: tatf tatc

tatf:
	mkdir -p $(FEAT_TAT)
	$(PYTHON) features.py -i $(SIM_TAT)/train.txt -o $(FEAT_TAT)/train.txt
	$(PYTHON) features.py -i $(SIM_TAT)/dev.txt -o $(FEAT_TAT)/dev.txt
	$(PYTHON) features.py -i $(SIM_TAT)/test-blind.txt -o $(FEAT_TAT)/test-blind.txt
	cp $(SIM_TAT)/labels.txt $(FEAT_TAT)/labels.txt

tatc:
	$(PYTHON) classify.py \
			--train $(FEAT_TAT)/train.txt \
			--dev $(FEAT_TAT)/dev.txt \
			--test $(FEAT_TAT)/test-blind.txt \
			--test_out $(OUT)/test-tat.txt \
			--labels $(FEAT_TAT)/labels.txt \
			--dev_iters 100000 \
			--train_max 1000000 --dev_max 100000 
			
tatcfast:
	$(PYTHON) classify.py \
			--train $(FEAT_TAT)/train.txt \
			--dev $(FEAT_TAT)/dev.txt \
			--test $(FEAT_TAT)/test-blind.txt \
			--test_out $(OUT)/test-tatfast.txt \
			--labels $(FEAT_TAT)/labels.txt \
			--dev_iters 200 \
			--train_max 1000 --dev_max 100 --test_max 100
			
prof:
	$(PYTHON) -m cProfile -s cumtime classify.py \
			--train $(FEAT_DSL)/train.txt \
			--dev $(FEAT_DSL)/dev.txt \
			--test $(FEAT_DSL)/test-blind.txt \
			--test_out $(OUT)/test-prof.txt \
			--labels $(FEAT_DSL)/labels.txt \
			--dev_iters 50000
	