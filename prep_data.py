import h5py
import json
import argparse


from utils import getSentencesMat , Vocabulary


parser = argparse.ArgumentParser(description='NMT Keras')
parser.add_argument('--text_A', type=str,  help='Corpus text file of language A')
parser.add_argument('--text_B', type=str,  help='Corpus text file of language B')
parser.add_argument('--out_file', type=str,  default="./data/nmt_hi_en_prepped.h5" , help='Output HDF5 file name')
args = parser.parse_args()


eng_sents = (open( args.text_A ).read()).split("\n")[:-1]
hi_sents = (open( args.text_B ).read()).split("\n")[:-1]


en_vocab = Vocabulary()
hi_vocab = Vocabulary()

for ens in eng_sents:
	en_vocab.add_words( ens.split(' ') )
	
for his in hi_sents:
	hi_vocab.add_words( his.split(' ') )

en_vocab.keepTopK(40000)
hi_vocab.keepTopK(40000)


eng_sent_mat = getSentencesMat( eng_sents  ,en_vocab , startEndTokens=True , tokenizer_fn=lambda x:x.split(' ') , maxSentenceL=100 )
hin_sent_mat = getSentencesMat( hi_sents  ,hi_vocab , startEndTokens=True , tokenizer_fn=lambda x:x.split(' ') , maxSentenceL=100 )



f = h5py.File( args.out_file , "w")

f.create_dataset("hi_vocab",   data=json.dumps(hi_vocab.getDicts()) )
f.create_dataset("en_vocab",   data=json.dumps(en_vocab.getDicts()) )

f.create_dataset("en_sents",   data=json.dumps( eng_sents ) )
f.create_dataset("hi_sents",   data=json.dumps( hi_sents ) )

f.create_dataset('eng_sent_mat' , data=eng_sent_mat )
f.create_dataset('hin_sent_mat' , data=hin_sent_mat )


f.close()


