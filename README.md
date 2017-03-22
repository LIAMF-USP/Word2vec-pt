# Word2vec-pt

This repository holds an implementation of the Skipgram model for training word embeddings. The corpus and the evaluation test are set up for Portuguese.



### Requirements
* Tensorflow
* Numpy
* Matplotlib

## Usage

```
$ python3 word2vec.py -h
usage: word2vec.py [-h] [-f FILE] [-s NUM_STEPS] [-v VOCAB_SIZE]
                   [-b BATCH_SIZE] [-e EMBED_SIZE] [-k SKIP_WINDOW]
                   [-n NUM_SKIPS] [-S NUM_SAMPLED] [-l LEARNING_RATE]
                   [-w SHOW_STEP] [-B VERBOSE_STEP] [-V VALID_SIZE]
                   [-W VALID_WINDOW]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  text file to apply the model (default=basic_pt.txt)
  -s NUM_STEPS, --num_steps NUM_STEPS
                        number of training steps (default=100000)
  -v VOCAB_SIZE, --vocab_size VOCAB_SIZE
                        vocab size (default=50000)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size (default=128)
  -e EMBED_SIZE, --embed_size EMBED_SIZE
                        embeddings size (default=128)
  -k SKIP_WINDOW, --skip_window SKIP_WINDOW
                        skip window (default=1)
  -n NUM_SKIPS, --num_skips NUM_SKIPS
                        number of skips, number of times a center word will be
                        re-used (default=2)
  -S NUM_SAMPLED, --num_sampled NUM_SAMPLED
                        number of negativ samples(default=64)
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate (default=1.0)
  -w SHOW_STEP, --show_step SHOW_STEP
                        show result in multiples of this step (default=2000)
  -B VERBOSE_STEP, --verbose_step VERBOSE_STEP
                        show similar words in multiples of this step
                        (default=10000)
  -V VALID_SIZE, --valid_size VALID_SIZE
                        number of words to display similarity(default=16)
  -W VALID_WINDOW, --valid_window VALID_WINDOW
                        number of words to from vocab to choose the words to
                        display similarity(default=100)


```


## Example

```
$ source download_wiki_pt.sh
$ python3 word2vec.py -f ./data/pt96.txt -s 2001 -w 200 -B 1000

&&&&&&&&& For TensorBoard visualization type &&&&&&&&&&&

tensorboard  --logdir=./graphs/22-03-2017_14-36-06


&&&&&&&&& And for the 3d embedding visualization type &&

tensorboard  --logdir=./processed

Nearest to apenas: Basco, Pocono, 820, Round, Medalha, cânones, Semanas, raparigas,
Nearest to e: ousava, Bardeen, Roadrunner, 1642, Tinha, amargura, garagem, meiotermo,
Nearest to foram: Epicuro, Ruído, manifestada, Duty, belos, Warwick, recémchegados, Destacamse,
Nearest to as: gênios, Joana, Nickelodeon, Duques, Trentino, Ptolomeu, resistir, dispensando,
Nearest to mesmo: nono, calorífico, Reinmuth, Dren, contasse, preocupado, 21ª, declararam,
Nearest to nos: botões, Nação, Raimundo, 1545, credores, Voltaire, Hizashi, n°,
Nearest to pode: Whatever, obtiveram, Pacto, software, bala, levante, escondia, partilhava,
Nearest to sua: 234, Féria, verossímil, sucessoras, Lin, cabra, estofada, ramificação,
Nearest to vez: Barragem, negligência, Azog, escolhe, Titan, analítica, enviada, relacionadas,
Nearest to O: contraditórias, Amizade, Bloomberg, folhagem, palcos, oriundo, ordenado, Irlanda,
Nearest to das: also, musica, 1510, cameo, templários, dedicam, Grão, festejado,
Nearest to Igreja: colaborador, vulcânicos, repensar, ocorrida, terrorista, restauracionistas, racista, verdadeiras,
Nearest to sem: Gill, Linnaeus, relutância, cortesã, Theatre, carnal, excedente, envolveuse,
Nearest to cidade: MaryClaire, interinstitucional, 564, adereços, adverte, circula, Wakulla·, inseparável,
Nearest to uma: marinhas, Olímpia, legítimos, 47, LP, Botan, bestseller, repúblicas,
Nearest to seus: mansões, dála, mosquito, Chipre•, Sociedade, Dolph, Tirion, colecionáveis,
Average loss at step 200 : 5.54808126211
Average loss at step 400 : 4.55995755911
Average loss at step 600 : 4.17230512619
Average loss at step 800 : 3.98659612417
Average loss at step 1000 : 4.02160902023
Nearest to apenas: Basco, 820, cânones, Pocono, 692, juridicamente, raparigas, Falava,
Nearest to e: UNK, amargura, Roadrunner, meiotermo, converteria, terceiro, Tinha, ousava,
Nearest to foram: Ruído, Epicuro, manifestada, Duty, belos, Warwick, Destacamse, Security,
Nearest to as: gênios, Excel, resistir, Trentino, Nickelodeon, vaginal, Joana, Duques,
Nearest to mesmo: deportadas, semanalmente, enfermo, pontifício, preocupado, Freedman·, civil, declararam,
Nearest to nos: botões, Nação, Raimundo, agreste, executivo, Blink, nostalgia, Renascimento,
Nearest to pode: Whatever, software, Pacto, bala, partilhava, Aranha, chuva, obtiveram,
Nearest to sua: 234, Féria, passavam, lho, ramificação, fiqh, temperamentos, esteve,
Nearest to vez: Barragem, negligência, Titan, Azog, enviada, relacionadas, escolhe, Iniciase,
Nearest to O: Amizade, Bloomberg, 1975·, contraditórias, folhagem, Neutron, EU, Karatedo,
Nearest to das: also, musica, templários, 1510, Kroemer, festejado, noivos, Bernhard,
Nearest to Igreja: colaborador, vulcânicos, repensar, ocorrida, terrorista, restauracionistas, racista, verdadeiras,
Nearest to sem: relutância, Gill, cortesã, Theatre, tomarem, porteiro, ambições, pontapé,
Nearest to cidade: interinstitucional, 564, 1810, adereços, adverte, Orleans, circula, góticas,
Nearest to uma: marinhas, a, Botan, bestseller, LP, Olímpia, CF, medidas,
Nearest to seus: mansões, dála, Chipre•, Rennais, Gelo, Sociedade, colecionáveis, firmas,
Average loss at step 1200 : 3.84192966342
Average loss at step 1400 : 3.78012679935
Average loss at step 1600 : 3.79607671261
Average loss at step 1800 : 3.78329489589
Average loss at step 2000 : 3.69324924827
Nearest to apenas: cânones, 820, 692, raparigas, juridicamente, Basco, Falava, DERSP,
Nearest to e: que, meiotermo, mas, Home, amargura, ou, 451, de,
Nearest to foram: Ruído, Epicuro, manifestada, Warwick, Duty, Destacamse, belos, imutável,
Nearest to as: Trentino, resistir, Nickelodeon, Excel, gênios, herética, Joana, vaginal,
Nearest to mesmo: enfermo, deportadas, pontifício, preocupado, Freedman·, semanalmente, que, civil,
Nearest to nos: botões, Nação, Renascimento, agreste, executivo, Raimundo, Blink, duquesa,
Nearest to pode: Whatever, partilhava, bala, software, devia, Aranha, Pacto, caseiros,
Nearest to sua: 234, lho, Féria, fiqh, melhoramento, passavam, esteve, shōji,
Nearest to vez: Barragem, negligência, Titan, Iniciase, enviada, Azog, relacionadas, gêmeos,
Nearest to O: o, Amizade, 1975·, Neutron, secretamente, Bloomberg, Karatedo, manso,
Nearest to das: also, festejado, musica, noivos, Bernhard, dedicam, Pontifícios, Kroemer,
Nearest to Igreja: colaborador, vulcânicos, repensar, ocorrida, terrorista, restauracionistas, racista, verdadeiras,
Nearest to sem: Kwan, ambições, relutância, Eu, Chicago, Gill, Kuroshio, receberem,
Nearest to cidade: interinstitucional, 564, 1810, adereços, Orleans, adverte, sans, Wakulla·,
Nearest to uma: a, marinhas, Botan, Progress, LP, deduz, CF, Stand,
Nearest to seus: mansões, dála, Chipre•, Gelo, Rennais, colecionáveis, Malaca, dançando,

==========================================

The emmbedding vectors can be found in
      ./pickles/pt96.pickle

```
