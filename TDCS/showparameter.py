from RNNmodels2 import RNNv1, neural, ADDv1
import pickle

f = open('TDCS/models/b_std_25_add.pkl', 'rb')
testneu = pickle.load(f)
f.close()

print(testneu.fuzzy_c)
print(testneu.fuzzy_sig)