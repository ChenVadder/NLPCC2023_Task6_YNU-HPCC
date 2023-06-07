#!/bin/bash
conda create NLPCCT6YNU
pip install -r requirements.txt
pip install sentence_transformers

python main.py
python 1-build_ES.py
python 2-get_temp_Top8.py
python 3-get_test_mg.py
python 4-sentence_transformers_Top8.py
python 5-get_entity_from_goldid.py
python 6-get_final_sub.py