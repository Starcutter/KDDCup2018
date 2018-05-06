clean:
	rm -rf data/*.pkl

rf:
	python utils/dataset.py
	python models/rf.py
	python utils/gen_csv.py
