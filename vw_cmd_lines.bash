# training oaa with embedding dictionary
vw --oaa 131 -b 24 -d WikiText2_cl100_e750_numex_max.vw --passes 1 -f model_WikiText2_cl100_e_750_numex_max --dictionary e:embeddings_vw.dict --ignore e --holdout_off