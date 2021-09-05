# training oaa with embedding dictionary
vw --oaa 131 -b 24 -d wikitext2_train_cl20_shuff_vw -c --passes 200 -f model.vw --dictionary e:embeddings_vw.dict --ignore e